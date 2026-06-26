"""Test helpers that avoid heavy package imports."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PKG_SRC = Path(__file__).resolve().parents[1] / "src"


def load_module_from_path(module_name: str, path: Path):
    """Load a module file without importing parent package ``__init__``."""
    if str(_PKG_SRC) not in sys.path:
        sys.path.insert(0, str(_PKG_SRC))
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def pkg_src() -> Path:
    return _PKG_SRC
