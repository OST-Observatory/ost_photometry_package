"""Structured logging with optional ANSI terminal formatting."""

from __future__ import annotations

import logging
import os
import sys

from . import style

_LOG = logging.getLogger("ost_photometry")
_CONFIGURED = False


class _AnsiFormatter(logging.Formatter):
  LEVEL_STYLES = {
      logging.DEBUG: style.Bcolors.NORMAL,
      logging.INFO: style.Bcolors.BOLD,
      logging.WARNING: style.Bcolors.WARNING,
      logging.ERROR: style.Bcolors.FAIL,
      logging.CRITICAL: style.Bcolors.FAIL,
  }

  def format(self, record: logging.LogRecord) -> str:
      prefix = self.LEVEL_STYLES.get(record.levelno, style.Bcolors.BOLD)
      return f"{prefix}{super().format(record)}{style.Bcolors.ENDC}"


class _TqdmAwareHandler(logging.StreamHandler):
  """Writes through ``tqdm.write`` so that messages appear above a running
  progress bar instead of inside its line."""

  def emit(self, record: logging.LogRecord) -> None:
      try:
          from tqdm import tqdm

          tqdm.write(self.format(record), file=self.stream)
          self.flush()
      except Exception:  # noqa: BLE001 - logging must not raise
          self.handleError(record)


def configure_logging(level: int | None = None) -> None:
    """Terminal logging; ``OST_PHOTOMETRY_VERBOSE=1`` also shows the
    per-frame details (``DETAIL`` messages)."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    if level is None:
        verbose = os.environ.get("OST_PHOTOMETRY_VERBOSE", "").strip().lower()
        level = logging.DEBUG if verbose in ("1", "true", "yes") else logging.INFO
    handler = _TqdmAwareHandler(sys.stdout)
    handler.setFormatter(_AnsiFormatter("%(message)s"))
    _LOG.addHandler(handler)
    _LOG.setLevel(level)
    # Keep CLI output on the package logger only; otherwise messages also
    # reach the root logger (e.g. from astropy) and appear twice.
    _LOG.propagate = False
    _CONFIGURED = True


def get_logger(name: str | None = None) -> logging.Logger:
    configure_logging()
    if name:
        return logging.getLogger(f"ost_photometry.{name}")
    return _LOG
