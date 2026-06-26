"""Structured logging with optional ANSI terminal formatting."""

from __future__ import annotations

import logging
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


def configure_logging(level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_AnsiFormatter("%(message)s"))
    _LOG.addHandler(handler)
    _LOG.setLevel(level)
    _CONFIGURED = True


def get_logger(name: str | None = None) -> logging.Logger:
    configure_logging()
    if name:
        return logging.getLogger(f"ost_photometry.{name}")
    return _LOG
