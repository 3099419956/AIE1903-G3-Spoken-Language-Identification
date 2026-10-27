"""Argument parsers shared by the command-line tool and local demo."""

import argparse
import os
from pathlib import Path


DEFAULT_MODEL = "facebook/mms-lid-126"


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def port_number(value: str) -> int:
    number = int(value)
    if not 0 <= number <= 65535:
        raise argparse.ArgumentTypeError("must be between 0 and 65535")
    return number


def build_predict_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predict.py",
        description="Identify spoken language with the course MMS-LID checkpoint.",
    )
    parser.add_argument("audio", nargs="+", type=Path, metavar="AUDIO")
    parser.add_argument(
        "--mode",
        choices=("whole", "pipeline"),
        default="whole",
        help="whole = official full-clip prediction; pipeline = checks, reject, and mixed speech",
    )
    parser.add_argument("--top-k", type=positive_int, default=5)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="write machine-readable JSON instead of the text report",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write JSON results to this file (creates the parent folder)",
    )
    return parser


def build_demo_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="demo_server.py",
        description="Serve the local MMS language identification demo.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=port_number, default=8765)
    parser.add_argument(
        "--threads",
        type=positive_int,
        default=min(8, os.cpu_count() or 4),
        help="PyTorch CPU threads (default: up to 8)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="open the demo URL after the model is ready",
    )
    return parser
