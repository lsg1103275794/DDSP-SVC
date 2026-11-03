import os
import time
import random
from contextlib import contextmanager, nullcontext
import numpy as np
import torch
import librosa
from logger.saver import Saver
from logger import utils
from torch.amp import autocast, GradScaler

def calculate_mel_snr(gt_mel, pred_mel):
    # 计算误差图像
    error_image = gt_mel - pred_mel
    # 计算参考图像的平方均值
    mean_square_reference = torch.mean(gt_mel ** 2)
    # 计算误差图像的方差
    variance_error = torch.var(error_image)
    # 计算并返回SNR
    snr = 10 * torch.log10(mean_square_reference / variance_error)
    return snr


def calculate_mel_si_snr(gt_mel, pred_mel):
    # 将测试图像按比例调整以最小化误差
    scale = torch.sum(gt_mel * pred_mel) / torch.sum(gt_mel ** 2)
    test_image_scaled = scale * gt_mel
    # 计算误差图像
    error_image = pred_mel - test_image_scaled 
    # 计算参考图像的平方均值
    mean_square_reference = torch.mean(gt_mel ** 2)
    # 计算误差图像的方差
    variance_error = torch.var(error_image)
    # 计算并返回SI-SNR
    si_snr = 10 * torch.log10(mean_square_reference / variance_error)
    return si_snr


def calculate_mel_psnr(gt_mel, pred_mel):
    # 计算误差图像
    error_image = gt_mel - pred_mel
    # 计算误差图像的均方误差
    mse = torch.mean(error_image ** 2)
    # 计算参考图像的最大可能功率
    max_power = torch.max(gt_mel) ** 2
    # 计算并返回PSNR
    psnr = 10 * torch.log10(max_power / mse)
    return psnr

def validation_mode_for_step(options, step):
    """Independent loss/preview/audit cadences; absent settings retain legacy audit."""
    mode = options.get('validation_mode', 'full_audit')
    if mode not in ('loss', 'preview', 'full_audit'):
        raise ValueError('validation_mode must be loss, preview or full_audit')
    for field, result in (('interval_audit', 'full_audit'),
                          ('interval_preview', 'preview'), ('interval_val', mode)):
        interval = options.get(field, 0)
        if interval and step % interval == 0:
            return result
    return None


@contextmanager
def validation_rng_context(device, policy='legacy', seed=0):
    if policy == 'legacy':
        yield
        return
    if policy != 'isolated':
        raise ValueError('validation_rng must be legacy or isolated')
    py_state, np_state = random.getstate(), np.random.get_state()
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(seed)
            yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def test(args, model, vocoder, loader_test, saver, mode=None):
    options = args.train
    mode = mode or options.get('validation_mode', 'full_audit')
    if mode not in ('loss', 'preview', 'full_audit'):
        raise ValueError('validation_mode must be loss, preview or full_audit')
    precision = options.get('validation_precision', 'fp32')
    dtypes = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}
    if precision not in dtypes:
        raise ValueError('validation_precision must be fp32, fp16 or bf16')
    with validation_rng_context(args.device, options.get('validation_rng', 'legacy'),
                                options.get('validation_seed', 0)):
        context = (nullcontext() if precision == 'fp32' else
                   autocast(device_type=torch.device(args.device).type, dtype=dtypes[precision]))
        with context:
            return _test(args, model, vocoder, loader_test, saver, mode)


def _test(args, model, vocoder, loader_test, saver, mode):
    print(' [*] testing...')
    model.eval()

    # losses
    test_ddsp_loss = 0.
    test_reflow_loss = 0.

    # mel mse val
    mel_val_mse_all = 0
    mel_val_mse_all_num = 0
    mel_val_snr_all = 0
    mel_val_psnr_all = 0
    mel_val_sisnr_all = 0

    # intialization
    num_batches = len(loader_test)
    if num_batches == 0:
        raise ValueError('validation loader is empty')
    preview_count = args.train.get('validation_preview_count', 2)
    if type(preview_count) is not int or preview_count < 0:
        raise ValueError('validation_preview_count must be a nonnegative integer')
    rtf_all = []
    spec_min = -6
    spec_max = 6
    spec_range = 12
    
    # run
    with torch.no_grad():
        for bidx, data in enumerate(loader_test):
            fn = data['name'][0]
            print('--------')
            print('{}/{} - {}'.format(bidx, num_batches, fn))

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            print('>>', data['name'][0])

            # Whole-set losses remain unchanged; only a deterministic prefix
            # synthesizes in preview mode. Full audit preserves call/RNG order.
            synthesize = mode == 'full_audit' or (mode == 'preview' and bidx < preview_count)
            if synthesize:
                st_time = time.time()
                mel = model(
                        data['units'], data['f0'], data['volume'], data['spk_id'],
                        vocoder=vocoder, infer=True, return_wav=False,
                        infer_step=args.infer.infer_step, method=args.infer.method,
                        t_start=args.model.t_start)
                signal = vocoder.infer(mel, data['f0'])
                ed_time = time.time()
                run_time = ed_time - st_time
                song_time = signal.shape[-1] / args.data.sampling_rate
                rtf = run_time / song_time
                print('RTF: {}  | {} / {}'.format(rtf, run_time, song_time))
                rtf_all.append(rtf)
           
            # loss
            ddsp_loss, reflow_loss = model(
                data['units'], 
                data['f0'], 
                data['volume'], 
                data['spk_id'],
                vocoder=vocoder,
                gt_spec=data['mel'],
                infer=False,
                t_start=args.model.t_start)
            test_ddsp_loss += ddsp_loss.item()
            test_reflow_loss += reflow_loss.item()
            
            if synthesize:
                saver.log_spec(data['name'][0], data['mel'], mel)
                path_audio = os.path.join(args.data.valid_path, 'audio', data['name_ext'][0])
                audio, sr = librosa.load(path_audio, sr=args.data.sampling_rate)
                if len(audio.shape) > 1:
                    audio = librosa.to_mono(audio)
                # TensorBoard consumes CPU audio; no unnecessary upload.
                audio = torch.from_numpy(audio).unsqueeze(0)
                saver.log_audio({fn+'/gt.wav': audio, fn+'/pred.wav': signal})

                # Legacy formulas intentionally retained for comparable full audits.
                mel_val_mse_all += torch.nn.functional.mse_loss(mel, data['mel']).detach().cpu().numpy()
                gt_mel_norm = torch.clip(data['mel'], spec_min, spec_max)
                gt_mel_norm = gt_mel_norm / spec_range + spec_min
                pre_mel_norm = torch.clip(mel, spec_min, spec_max)
                pre_mel_norm = pre_mel_norm / spec_range + spec_min
                mel_val_snr_all += calculate_mel_snr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
                mel_val_psnr_all += calculate_mel_psnr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
                mel_val_sisnr_all += calculate_mel_si_snr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
                mel_val_mse_all_num += 1
            
    # report
    test_ddsp_loss /= num_batches
    test_reflow_loss /= num_batches 
    if not mel_val_mse_all_num:
        print(' [validation] loss only; no synthesis metrics or artifacts')
        return test_ddsp_loss, test_reflow_loss
    mel_val_mse_all /= mel_val_mse_all_num
    mel_val_snr_all /= mel_val_mse_all_num
    mel_val_psnr_all /= mel_val_mse_all_num
    mel_val_sisnr_all /= mel_val_mse_all_num

    # check
    print(' [test_ddsp_loss] test_ddsp_loss:', test_ddsp_loss)
    print(' [test_reflow_loss] test_reflow_loss:', test_reflow_loss)
    print(' Real Time Factor', np.mean(rtf_all))
    metric_prefix = 'validation' if mode == 'full_audit' else 'validation_preview'
    print(f' [validation] synthesis metrics scope={mode}, clips={mel_val_mse_all_num}')
    print(' Mel Val MSE', mel_val_mse_all)
    saver.log_value({
        metric_prefix+'/mel_val_mse': mel_val_mse_all
    })
    print(' Mel Val SNR', mel_val_snr_all)
    saver.log_value({
        metric_prefix+'/mel_val_snr': mel_val_snr_all
    })
    print(' Mel Val PSNR', mel_val_psnr_all)
    saver.log_value({
        metric_prefix+'/mel_val_psnr': mel_val_psnr_all
    })
    print(' Mel Val SI-SNR', mel_val_sisnr_all)
    saver.log_value({
        metric_prefix+'/mel_val_sisnr': mel_val_sisnr_all
    })
    return test_ddsp_loss, test_reflow_loss


def train(args, initial_global_step, model, optimizer, scheduler, vocoder, loader_train, loader_test):
    # saver
    saver = Saver(args, initial_global_step=initial_global_step)

    # model size
    params_count = utils.get_network_paras_amount({'model': model})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)
    
    # run
    num_batches = len(loader_train)
    start_epoch = initial_global_step // num_batches
    model.train()
    saver.log_info('======= start training =======')
    scaler = GradScaler()
    if args.train.amp_dtype == 'fp32':
        dtype = torch.float32
    elif args.train.amp_dtype == 'fp16':
        dtype = torch.float16
    elif args.train.amp_dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError(' [x] Unknown amp_dtype: ' + args.train.amp_dtype)
    for epoch in range(start_epoch, args.train.epochs):
        for batch_idx, data in enumerate(loader_train):
            saver.global_step_increment()
            optimizer.zero_grad()

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            
            # forward
            if dtype == torch.float32:
                ddsp_loss, reflow_loss = model(data['units'].float(), data['f0'], data['volume'], data['spk_id'], 
                                aug_shift=data['aug_shift'], vocoder=vocoder, gt_spec=data['mel'].float(), infer=False, t_start=args.model.t_start)
            else:
                with autocast(device_type=args.device, dtype=dtype):
                    ddsp_loss, reflow_loss=model(data['units'], data['f0'], data['volume'], data['spk_id'], 
                                    aug_shift=data['aug_shift'], vocoder=vocoder, gt_spec=data['mel'].float(), infer=False, t_start=args.model.t_start)
            
            # handle nan loss
            if torch.isnan(ddsp_loss):
                print(' [x] nan ddsp_loss ')
                optimizer.zero_grad()
                del ddsp_loss
                del reflow_loss
                continue
            elif torch.isnan(reflow_loss):
                raise ValueError(' [x] nan reflow_loss ')
            else:
                loss = args.train.lambda_ddsp * ddsp_loss + reflow_loss
                # backpropagate
                if dtype == torch.float32:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                scheduler.step()
                
            # log loss
            if saver.global_step % args.train.interval_log == 0:
                current_lr =  optimizer.param_groups[0]['lr']
                saver.log_info(
                    'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | lr: {:.6} | loss: {:.3f} | time: {} | step: {}'.format(
                        epoch,
                        batch_idx,
                        num_batches,
                        args.env.expdir,
                        args.train.interval_log/saver.get_interval_time(),
                        current_lr,
                        loss.item(),
                        saver.get_total_time(),
                        saver.global_step
                    )
                )
                
                saver.log_value({
                    'train/loss': loss.item(),
                    'train/ddsp_loss': ddsp_loss.item(),
                    'train/reflow_loss': reflow_loss.item(),
                    'train/lr': current_lr
                })
            
            # validation
            validation_mode = validation_mode_for_step(args.train, saver.global_step)
            if validation_mode is not None:
                optimizer_save = optimizer if args.train.save_opt else None
                
                # Keep existing save cadence independent of optional previews/audits.
                if saver.global_step % args.train.interval_val == 0:
                    saver.save_model(model, optimizer_save, postfix=f'{saver.global_step}')
                    last_val_step = saver.global_step - args.train.interval_val
                    if last_val_step % args.train.interval_force_save != 0:
                        saver.delete_model(postfix=f'{last_val_step}')
                
                # run testing set
                test_ddsp_loss, test_reflow_loss = test(args, model, vocoder, loader_test, saver, mode=validation_mode)
                test_loss = args.train.lambda_ddsp * test_ddsp_loss + test_reflow_loss
                
                # log loss
                saver.log_info(
                    ' --- <validation> --- \nloss: {:.3f}. '.format(
                        test_loss,
                    )
                )
                
                saver.log_value({
                    'validation/loss': test_loss,
                    'validation/ddsp_loss': test_ddsp_loss,
                    'validation/reflow_loss': test_reflow_loss
                })
                
                model.train()
