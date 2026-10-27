import io
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from scripts.cli_options import build_demo_parser, build_predict_parser


class PredictParserTests(unittest.TestCase):
    def test_accepts_multiple_audio_files_and_pipeline_mode(self) -> None:
        args = build_predict_parser().parse_args(
            ["--mode", "pipeline", "--json", "--top-k", "3", "a.wav", "b.wav"]
        )

        self.assertEqual(args.audio, [Path("a.wav"), Path("b.wav")])
        self.assertEqual(args.mode, "pipeline")
        self.assertTrue(args.as_json)
        self.assertEqual(args.top_k, 3)

    def test_rejects_zero_threads(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_predict_parser().parse_args(["--threads", "0", "a.wav"])


class DemoParserTests(unittest.TestCase):
    def test_accepts_custom_host_and_port(self) -> None:
        args = build_demo_parser().parse_args(
            ["--host", "0.0.0.0", "--port", "9000", "--open-browser"]
        )

        self.assertEqual(args.host, "0.0.0.0")
        self.assertEqual(args.port, 9000)
        self.assertTrue(args.open_browser)

    def test_rejects_invalid_port(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_demo_parser().parse_args(["--port", "70000"])


if __name__ == "__main__":
    unittest.main()
