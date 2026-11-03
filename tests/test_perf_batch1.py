"""CPU regressions for the transparent first-batch optimizations.
Run with the DDSP interpreter: python -I -X utf8 -B <absolute test path>.
"""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from nsf_hifigan.nvSTFT import STFT


class WindowCacheTests(unittest.TestCase):
    def test_same_rounded_geometry_reuses_window_and_mel_is_exact(self):
        stft = STFT(sr=44100, n_mels=128, n_fft=2048, win_size=2048,
                    hop_length=512, fmax=22050)
        audio = torch.linspace(-0.1, 0.1, 4096).unsqueeze(0)
        first = stft.get_mel(audio, keyshift=1.0)
        window = next(iter(stft.hann_window.values()))
        second = stft.get_mel(audio, keyshift=1.00001)
        self.assertEqual(len(stft.hann_window), 1)
        self.assertIs(window, next(iter(stft.hann_window.values())))
        self.assertTrue(torch.equal(first, second))
        length = int(np.round(2048 * 2 ** (1.0 / 12)))
        self.assertTrue(torch.equal(window, torch.hann_window(length)))

    def test_cache_is_bounded_and_dtype_sensitive(self):
        stft = STFT(window_cache_size=4)
        for length in range(100, 110):
            stft._get_hann_window(length, torch.device('cpu'))
        self.assertEqual(len(stft.hann_window), 4)
        first = stft._get_hann_window(109, torch.device('cpu'))
        old_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            second = stft._get_hann_window(109, torch.device('cpu'))
        finally:
            torch.set_default_dtype(old_dtype)
        self.assertEqual(first.dtype, torch.float32)
        self.assertEqual(second.dtype, torch.float64)
        self.assertEqual(len(stft.hann_window), 4)

    def test_cache_rejects_invalid_bound(self):
        for size in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                STFT(window_cache_size=size)


class GeometryTests(unittest.TestCase):
    @staticmethod
    def model(**overrides):
        from ddsp.vocoder import CombSubSuperFast
        options = dict(sampling_rate=44100, block_size=32, win_length=128,
                       n_unit=8, num_layers=1, dim_model=32)
        options.update(overrides)
        return CombSubSuperFast(**options)

    def test_integer_metadata_and_legacy_state_keys(self):
        model = self.model()
        self.assertEqual((model.sampling_rate_int, model.block_size_int,
                          model.win_length_int), (44100, 32, 128))
        self.assertTrue({'sampling_rate', 'block_size', 'win_length', 'window'}
                        <= model.state_dict().keys())
        model.load_state_dict(model.state_dict(), strict=True)
        with self.assertRaises(AttributeError):
            model.block_size_int = 64

    def test_inconsistent_checkpoint_geometry_fails_even_non_strict(self):
        for key in ('sampling_rate', 'block_size', 'win_length'):
            model = self.model()
            state = model.state_dict()
            state[key] = state[key] + 1
            with self.assertRaisesRegex(RuntimeError, 'geometry mismatch'):
                model.load_state_dict(state, strict=False)

    def test_invalid_configuration_geometry_fails(self):
        for value in (0, -1, 32.5, True):
            with self.assertRaises(ValueError):
                self.model(block_size=value)


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_policy_is_explicit_and_cpu_staged(self):
        from logger.checkpoint import load_checkpoint
        with mock.patch('torch.load', return_value={'model': {}}) as load:
            load_checkpoint('fixture.pt')
            self.assertEqual(load.call_args.kwargs,
                             dict(map_location='cpu', weights_only=False))
            load_checkpoint('fixture.pt', checkpoint_policy='weights_only')
            self.assertTrue(load.call_args.kwargs['weights_only'])
        with self.assertRaises(ValueError):
            load_checkpoint('fixture.pt', checkpoint_policy='automatic')

    def test_svc_load_is_strict_and_loads_before_device_transfer(self):
        from reflow import vocoder
        events = []
        model = mock.Mock()
        model.load_state_dict.side_effect = lambda *a, **kw: events.append(('load', kw))
        model.to.side_effect = lambda *a, **kw: events.append(('to', a))
        config = dict(vocoder=dict(type='nsf-hifigan', ckpt='fixture'),
                      model=dict(type='RectifiedFlow'), data={})
        with mock.patch('builtins.open', mock.mock_open()), \
             mock.patch.object(vocoder.yaml, 'safe_load', return_value=config), \
             mock.patch.object(vocoder, 'Vocoder') as voc, \
             mock.patch.object(vocoder, 'Unit2Wav', return_value=model), \
             mock.patch.object(vocoder, 'load_checkpoint', return_value={'model': {}}):
            voc.return_value.dimension = 128
            vocoder.load_model_vocoder('fixture/model.pt', device='cuda:0')
        self.assertEqual(events, [('load', {'strict': True}), ('to', ('cuda:0',))])


class ValidationTests(unittest.TestCase):
    def test_independent_cadences_keep_legacy_default(self):
        from reflow.solver import validation_mode_for_step
        self.assertEqual(validation_mode_for_step({'interval_val': 10}, 10), 'full_audit')
        options = dict(interval_val=10, validation_mode='loss',
                       interval_preview=25, interval_audit=100)
        self.assertIsNone(validation_mode_for_step(options, 1))
        self.assertEqual(validation_mode_for_step(options, 10), 'loss')
        self.assertEqual(validation_mode_for_step(options, 25), 'preview')
        self.assertEqual(validation_mode_for_step(options, 100), 'full_audit')

    def test_isolated_rng_restores_python_numpy_and_torch(self):
        from reflow.solver import validation_rng_context
        import random
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        py = random.getstate()
        np_state = np.random.get_state()
        state = torch.get_rng_state().clone()
        with validation_rng_context('cpu', 'isolated', seed=123):
            random.random()
            np.random.rand()
            torch.rand(3)
        self.assertEqual(random.getstate(), py)
        self.assertTrue(np.array_equal(np.random.get_state()[1], np_state[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), state))

    def test_loss_only_does_not_synthesize_or_log_artifacts(self):
        from reflow.solver import test
        from logger.utils import DotDict
        args = DotDict(dict(device='cpu', train={}, model=dict(t_start=0.0)))
        model = mock.Mock(return_value=(torch.tensor(1.), torch.tensor(2.)))
        data = dict(name=['clip'], units=torch.zeros(1, 2, 8),
                    f0=torch.ones(1, 2, 1), volume=torch.ones(1, 2, 1),
                    spk_id=torch.ones(1, 1), mel=torch.zeros(1, 2, 128))
        vocoder, saver = mock.Mock(), mock.Mock()
        result = test(args, model, vocoder, [data], saver, mode='loss')
        self.assertEqual(result, (1., 2.))
        self.assertEqual(model.call_count, 1)
        vocoder.infer.assert_not_called()
        saver.log_audio.assert_not_called()
        saver.log_spec.assert_not_called()

    def test_preview_is_deterministic_subset_and_metrics_are_labeled(self):
        from reflow.solver import test
        from logger.utils import DotDict
        args = DotDict(dict(device='cpu', train=dict(validation_preview_count=1),
                            model=dict(t_start=0.), infer=dict(infer_step=50, method='euler'),
                            data=dict(sampling_rate=44100, valid_path='fixture')))
        rows = [dict(name=[f'clip{i}'], name_ext=[f'clip{i}.wav'],
                     units=torch.zeros(1, 2, 8), f0=torch.ones(1, 2, 1),
                     volume=torch.ones(1, 2, 1), spk_id=torch.ones(1, 1),
                     mel=torch.ones(1, 2, 128)) for i in range(3)]
        model = mock.Mock(side_effect=lambda *a, **kw: torch.full((1, 2, 128), .5)
                          if kw['infer'] else (torch.tensor(1.), torch.tensor(2.)))
        vocoder, saver = mock.Mock(), mock.Mock()
        vocoder.infer.return_value = torch.ones(1, 1024)
        with mock.patch('reflow.solver.librosa.load', return_value=(np.ones(1024, dtype=np.float32), 44100)) as load:
            self.assertEqual(test(args, model, vocoder, rows, saver, mode='preview'), (1., 2.))
        self.assertEqual(model.call_count, 4)
        self.assertEqual(vocoder.infer.call_count, 1)
        self.assertEqual(load.call_count, 1)
        self.assertEqual(saver.log_spec.call_args.args[0], 'clip0')
        self.assertTrue(all(key.startswith('validation_preview/')
                            for call in saver.log_value.call_args_list for key in call.args[0]))
        self.assertEqual(saver.log_audio.call_args.args[0]['clip0/gt.wav'].device.type, 'cpu')


class UpdateTests(unittest.TestCase):
    def test_bf16_has_no_scaler_and_counts_only_finite_updates(self):
        from reflow.solver import OptimizationStep
        parameter = torch.nn.Parameter(torch.tensor([1.]))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        step = OptimizationStep(optimizer, scheduler, 'cpu', torch.bfloat16)
        self.assertIsNone(step.scaler)
        parameter.register_hook(lambda gradient: torch.full_like(gradient, float('inf')))
        self.assertFalse(step.step(parameter.square().sum()))
        self.assertTrue(torch.equal(parameter, torch.tensor([1.])))
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.1)
        self.assertEqual((step.attempted, step.successful, step.skipped), (1, 0, 1))

    def test_finite_step_and_nonfinite_loss_accounting(self):
        from reflow.solver import OptimizationStep
        parameter = torch.nn.Parameter(torch.tensor([1.]))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        step = OptimizationStep(optimizer, scheduler, 'cpu', torch.float32)
        self.assertTrue(step.step(parameter.square().sum()))
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.05)
        snapshot = parameter.detach().clone()
        optimizer.zero_grad()
        self.assertFalse(step.step(parameter.sum() * float('inf')))
        self.assertTrue(torch.equal(parameter, snapshot))
        self.assertEqual((step.attempted, step.successful, step.skipped), (2, 1, 1))

    def test_chained_optimizer_never_partially_updates_on_overflow(self):
        from reflow.solver import OptimizationStep
        from optimizer.chained_optimizer import ChainedOptimizer, OptimizerSpec
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            weight = torch.nn.Parameter(torch.tensor([[1.]]))
            bias = torch.nn.Parameter(torch.tensor([1.]))
            optimizer = ChainedOptimizer([weight, bias], [
                OptimizerSpec(torch.optim.SGD, {}, lambda p: p.ndim == 2),
                OptimizerSpec(torch.optim.AdamW, {}, None)], lr=0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.5)
            step = OptimizationStep(optimizer, scheduler, 'cpu', dtype)
            self.assertTrue(step.step(weight.square().sum() + bias.square().sum()))
            snapshot = [p.detach().clone() for p in (weight, bias)]
            epoch, lr = scheduler.last_epoch, optimizer.param_groups[0]['lr']
            optimizer.zero_grad()
            handle = bias.register_hook(lambda g: torch.full_like(g, float('nan')))
            self.assertFalse(step.step(weight.square().sum() + bias.square().sum()))
            handle.remove()
            self.assertTrue(all(torch.equal(p, s) for p, s in zip((weight, bias), snapshot)))
            self.assertEqual((scheduler.last_epoch, optimizer.param_groups[0]['lr']), (epoch, lr))
            self.assertEqual((step.attempted, step.successful, step.skipped), (2, 1, 1))

    def test_fp16_scaler_skips_inf_and_resumes_counter_state(self):
        from reflow.solver import OptimizationStep
        parameter = torch.nn.Parameter(torch.tensor([1.]))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        step = OptimizationStep(optimizer, scheduler, 'cpu', torch.float16)
        self.assertIsNotNone(step.scaler)
        parameter.register_hook(lambda gradient: gradient * float('inf'))
        self.assertFalse(step.step(parameter.square().sum()))
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.1)
        state = step.state_dict()
        replacement = OptimizationStep(optimizer, scheduler, 'cpu', torch.float16)
        replacement.load_state_dict(state)
        self.assertEqual(replacement.state_dict(), state)


class TrainingControlTests(unittest.TestCase):
    def test_saver_counter_advances_after_updates_not_attempts(self):
        from reflow import solver
        from logger.utils import DotDict
        parameter = torch.nn.Parameter(torch.tensor([1.]))
        optimizer = torch.optim.SGD([parameter], lr=.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.5)
        rows = [dict(units=torch.tensor(float(i))) for i in range(3)]

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.parameter = parameter
            def forward(self, units, *a, **kw):
                loss = self.parameter.square().sum()
                if int(units.item()) == 1:
                    loss = loss * float('inf')
                return loss, loss * 0

        class Saver:
            def __init__(self, *a, **kw): self.global_step=0; self.values=[]
            def global_step_increment(self): self.global_step+=1
            def log_info(self, *a): pass
            def log_value(self, values): self.values.append(values)
            def get_interval_time(self): return 1
            def get_total_time(self): return 1

        saver = Saver()
        for row in rows:
            row.update(f0=torch.zeros(1), volume=torch.zeros(1), spk_id=torch.ones(1),
                       aug_shift=torch.zeros(1), mel=torch.zeros(1))
        args = DotDict(dict(device='cpu', env=dict(expdir='fixture'),
                            model=dict(t_start=0.), train=dict(amp_dtype='bf16',epochs=1,
                            lambda_ddsp=1.,interval_log=1,interval_val=100)))
        with mock.patch.object(solver, 'Saver', return_value=saver):
            solver.train(args, 0, Model(), optimizer, scheduler, None, rows, [])
        self.assertEqual(saver.global_step, 2)
        self.assertEqual(scheduler.last_epoch, 2)
        self.assertEqual(saver.values[-1]['train/attempted_steps'], 3)
        self.assertEqual(saver.values[-1]['train/skipped_steps'], 1)


if __name__ == '__main__':
    unittest.main()
