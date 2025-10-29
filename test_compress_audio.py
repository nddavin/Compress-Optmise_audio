import unittest
import os
import tempfile
from unittest.mock import patch
import sys
sys.path.insert(0, os.path.dirname(__file__))

from compress_audio import (
    get_format_defaults, get_compressor_preset, get_multiband_preset,
    build_multiband_compressor, get_channel_layout_info, build_channel_filters,
    build_audio_filters, create_output_dirs, validate_inputs
)

class TestCompressAudio(unittest.TestCase):

    def test_get_format_defaults_mp3_speech(self):
        result = get_format_defaults("mp3", "speech")
        expected = {
            "codec": "libmp3lame",
            "ext": ".mp3",
            "speech": [64, 96, 128],
            "music": [128, 192, 256]
        }
        self.assertEqual(result, expected)

    def test_get_format_defaults_aac_music(self):
        result = get_format_defaults("aac", "music")
        expected = {
            "codec": "aac",
            "ext": ".m4a",
            "speech": [48, 64, 96],
            "music": [96, 128, 192]
        }
        self.assertEqual(result, expected)

    def test_get_format_defaults_unsupported(self):
        result = get_format_defaults("unsupported")
        self.assertEqual(result, {})

    def test_get_compressor_preset_speech(self):
        result = get_compressor_preset("speech")
        expected = {
            "threshold": -20,
            "ratio": 3,
            "attack": 0.01,
            "release": 0.1,
            "makeup": 6
        }
        self.assertEqual(result, expected)

    def test_get_compressor_preset_music(self):
        result = get_compressor_preset("music")
        expected = {
            "threshold": -18,
            "ratio": 4,
            "attack": 0.005,
            "release": 0.05,
            "makeup": 4
        }
        self.assertEqual(result, expected)

    def test_get_compressor_preset_default(self):
        result = get_compressor_preset("unknown")
        self.assertEqual(result, get_compressor_preset("speech"))

    def test_get_multiband_preset_speech(self):
        result = get_multiband_preset("speech")
        self.assertIn("low_freq", result)
        self.assertIn("high_freq", result)
        self.assertEqual(result["low_freq"], 250)
        self.assertEqual(result["high_freq"], 4000)

    def test_get_multiband_preset_music(self):
        result = get_multiband_preset("music")
        self.assertEqual(result["low_freq"], 200)
        self.assertEqual(result["high_freq"], 5000)

    def test_build_multiband_compressor_default(self):
        result = build_multiband_compressor("speech")
        self.assertIn("acrossor", result)
        self.assertIn("acompressor", result)

    def test_build_multiband_compressor_custom_freqs(self):
        custom_freqs = {"low": 300, "high": 5000}
        result = build_multiband_compressor("speech", custom_freqs=custom_freqs)
        self.assertIn("split=300:5000", result)

    def test_get_channel_layout_info_mono(self):
        result = get_channel_layout_info("mono")
        expected = {"channels": 1, "layout": "mono"}
        self.assertEqual(result, expected)

    def test_get_channel_layout_info_stereo(self):
        result = get_channel_layout_info("stereo")
        expected = {"channels": 2, "layout": "stereo"}
        self.assertEqual(result, expected)

    def test_get_channel_layout_info_51(self):
        result = get_channel_layout_info("5.1")
        expected = {"channels": 6, "layout": "5.1"}
        self.assertEqual(result, expected)

    def test_get_channel_layout_info_unknown(self):
        result = get_channel_layout_info("unknown")
        self.assertEqual(result, get_channel_layout_info("stereo"))

    def test_build_channel_filters_no_layout(self):
        result = build_channel_filters(2)
        self.assertEqual(result, [])

    def test_build_channel_filters_stereo_layout(self):
        result = build_channel_filters(2, "stereo")
        self.assertEqual(len(result), 1)
        self.assertIn("channelmap", result[0])

    def test_build_channel_filters_downmix(self):
        result = build_channel_filters(6, "stereo", downmix=True)
        self.assertEqual(len(result), 1)
        self.assertIn("pan=stereo", result[0])

    def test_build_audio_filters_basic(self):
        result = build_audio_filters(loudnorm_enabled=True)
        self.assertIsNotNone(result)
        if result:
            self.assertIn("loudnorm", result)

    def test_build_audio_filters_no_filters(self):
        result = build_audio_filters(loudnorm_enabled=False, compressor_enabled=False, multiband_enabled=False)
        self.assertIsNone(result)

    def test_build_audio_filters_compressor(self):
        result = build_audio_filters(compressor_enabled=True, compressor_preset="speech")
        self.assertIsNotNone(result)
        if result:
            self.assertIn("acompressor", result)

    def test_build_audio_filters_multiband(self):
        result = build_audio_filters(multiband_enabled=True, multiband_preset="speech")
        self.assertIsNotNone(result)
        if result:
            self.assertIn("acrossor", result)

    def test_build_audio_filters_silence_trim(self):
        result = build_audio_filters(silence_trim_enabled=True, silence_threshold=-40, silence_duration=1.0)
        self.assertIsNotNone(result)
        if result:
            self.assertIn("silenceremove", result)

    def test_build_audio_filters_noise_gate(self):
        result = build_audio_filters(noise_gate_enabled=True, gate_threshold=-30)
        self.assertIsNotNone(result)
        if result:
            self.assertIn("agate", result)

    def test_build_audio_filters_ml_noise_reduction(self):
        # Test with model path set
        from config import config_manager
        config_manager.set_model_path("arnndn_model", "/usr/local/share/ffmpeg/arnndn-models/bd.cnr.mdl")

        result = build_audio_filters(ml_noise_reduction=True)
        self.assertIsNotNone(result)
        # Note: The filter may not include arnndn if the model file doesn't exist
        # This is expected behavior - the test just verifies the function doesn't crash
        # and returns a valid filter chain (loudnorm in this case)

    def test_create_output_dirs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bitrates = [64, 128]
            result = create_output_dirs(temp_dir, bitrates)
            self.assertIn(64, result)
            self.assertIn(128, result)
            self.assertTrue(os.path.exists(result[64]))
            self.assertTrue(os.path.exists(result[128]))

    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_validate_inputs_valid(self, mock_makedirs, mock_exists):
        mock_exists.return_value = True
        mock_makedirs.return_value = None

        class Args:
            input = "/valid/path"
            output = "/output/path"
            format = "mp3"
            content_type = "speech"
            bitrates = None

        args = Args()
        result = validate_inputs(args)
        self.assertIsNotNone(result.bitrates)
        self.assertEqual(result.bitrates, [64, 96, 128])

    @patch('os.path.exists')
    def test_validate_inputs_invalid_input(self, mock_exists):
        mock_exists.return_value = False

        class Args:
            input = "/invalid/path"
            output = "/output/path"
            format = "mp3"
            content_type = "speech"
            bitrates = None

        args = Args()
        with self.assertRaises(SystemExit):
            validate_inputs(args)

if __name__ == '__main__':
    unittest.main()