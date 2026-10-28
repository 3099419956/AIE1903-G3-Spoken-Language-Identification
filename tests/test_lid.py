import unittest

import numpy as np

from scripts.lid import add_noise, crop_speech, judge, window_slices


class JudgeTests(unittest.TestCase):
    def test_single_language_hit(self):
        self.assertEqual(judge("cmn", "cmn"), "HIT")

    def test_wrong_language_is_miss(self):
        self.assertEqual(judge("cmn", "eng"), "MISS")

    def test_uncertain_is_reject(self):
        self.assertEqual(judge("cmn", "uncertain"), "REJECT")

    def test_mixed_needs_both_languages(self):
        self.assertEqual(judge("mixed", "mixed:cmn+eng"), "HIT")
        self.assertEqual(judge("mixed", "mixed:cmn"), "PARTIAL")
        self.assertEqual(judge("mixed", "eng"), "PARTIAL")
        self.assertEqual(judge("mixed", "kor"), "MISS")


class WindowSlicesTests(unittest.TestCase):
    def test_short_audio_is_one_window(self):
        self.assertEqual(window_slices(8000, 16000, 8000), [(0, 8000)])

    def test_two_windows_for_1_5_seconds(self):
        self.assertEqual(window_slices(24000, 16000, 8000), [(0, 16000), (8000, 24000)])


class CropSpeechTests(unittest.TestCase):
    def test_crop_pads_when_too_short(self):
        x = np.zeros(8000, dtype=np.float32)
        self.assertEqual(crop_speech(x, 1.0, start=0).shape[0], 16000)

    def test_crop_takes_requested_length(self):
        x = np.random.default_rng(0).normal(size=32000).astype(np.float32)
        self.assertEqual(crop_speech(x, 1.0, start=0).shape[0], 16000)


class AddNoiseTests(unittest.TestCase):
    def test_noise_preserves_shape_and_is_finite(self):
        x = np.random.default_rng(1).normal(size=16000).astype(np.float32)
        y = add_noise(x, 10.0, rng=np.random.default_rng(0))
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(np.isfinite(y).all())


if __name__ == "__main__":
    unittest.main()
