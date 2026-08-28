"""
Logging utilities for the MLLM uncertainty decomposition project.

Usage:
    from src.utils.logging_utils import get_logger, setup_logging

    setup_logging(log_file="results/run.log", level=logging.INFO)
    logger = get_logger(__name__)
    logger.info("Starting experiment")
"""

import json
import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    also_stdout: bool = True,
) -> None:
    """
    Configure root logger with optional file and stdout handlers.

    Args:
        log_file: Path to log file. If None, logs to stdout only.
        level: Logging level (e.g. logging.INFO, logging.DEBUG).
        also_stdout: If True and log_file is set, also log to stdout.
    """
    formatter = logging.Formatter(fmt=_FMT, datefmt=_DATE_FMT)
    root = logging.getLogger()
    root.setLevel(level)

    # Clear existing handlers to avoid duplicate logs on re-initialization
    root.handlers.clear()

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(formatter)
        root.addHandler(fh)

    if also_stdout or log_file is None:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        root.addHandler(sh)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger. Call setup_logging() once at startup."""
    return logging.getLogger(name)


class ExperimentLogger:
    """
    Structured experiment logger that keeps a running JSON-lines log
    of each sample's outputs alongside the standard text log.

    Use this to record inference results in a recoverable, append-safe format.
    """

    def __init__(self, output_path: str, experiment_name: str = ""):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self._log = get_logger(f"experiment.{experiment_name}")
        self._file = open(self.output_path, "a", encoding="utf-8")
        self._log.info(
            f"ExperimentLogger initialized: {self.output_path} "
            f"(experiment={experiment_name!r})"
        )

    def log_sample(self, sample_dict: dict) -> None:
        """Append one sample's results as a JSON line."""
        sample_dict["_logged_at"] = datetime.utcnow().isoformat()
        sample_dict["_experiment"] = self.experiment_name
        self._file.write(json.dumps(sample_dict, ensure_ascii=False) + "\n")
        self._file.flush()

    def log_summary(self, summary: dict) -> None:
        """Log a summary dict at the end of a run."""
        self._log.info(f"Experiment summary: {json.dumps(summary, indent=2)}")
        summary_path = self.output_path.with_suffix(".summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        self._log.info(f"Summary saved to {summary_path}")

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
