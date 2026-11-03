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


if __name__ == '__main__':
    unittest.main()
