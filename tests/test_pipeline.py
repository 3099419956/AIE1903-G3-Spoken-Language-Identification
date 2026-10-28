import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pipeline


class SpeechBoundsTests(unittest.TestCase):
    def test_silence_has_no_bounds(self):
        self.assertIsNone(pipeline.speech_bounds(np.zeros(16000, dtype=np.float32)))

    def test_tone_has_bounds_inside_audio(self):
        t = np.arange(32000, dtype=np.float32) / 16000.0
        tone = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        start, end = pipeline.speech_bounds(tone)
        self.assertLess(start, end)
        self.assertLessEqual(end, len(tone))


class SnrTests(unittest.TestCase):
    @staticmethod
    def _tone_burst():
        x = np.zeros(32000, dtype=np.float32)
        t = np.arange(32000, dtype=np.float32) / 16000.0
        tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        x[10000:20000] = tone[10000:20000]
        return x

    def test_clean_speech_is_cleaner_than_noisy(self):
        rng = np.random.default_rng(0)
        x = self._tone_burst()
        noisy = x + rng.normal(0, 0.1, size=x.shape).astype(np.float32)
        self.assertGreater(pipeline.estimate_snr_db(x), pipeline.estimate_snr_db(noisy))


class Mono16kTests(unittest.TestCase):
    def test_stereo_downmixes_to_mono(self):
        stereo = np.random.default_rng(0).normal(size=(16000, 2)).astype(np.float32)
        x, _ = pipeline.to_mono_16k(stereo, 16000)
        self.assertEqual(x.ndim, 1)
        self.assertEqual(x.shape[0], 16000)

    def test_cap_at_eight_seconds(self):
        long_audio = np.zeros(16000 * 10, dtype=np.float32)
        x, _ = pipeline.to_mono_16k(long_audio, 16000)
        self.assertEqual(x.shape[0], 16000 * 8)


if __name__ == "__main__":
    unittest.main()
