"""Test helpers that avoid heavy package imports."""

from __future__ import annotations

import importlib.util
import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_PKG_SRC = Path(__file__).resolve().parents[1] / "src"


@contextmanager
def isolated_sys_modules() -> Iterator[None]:
    """Restore ``sys.modules`` after a test that registers import stubs."""
    before = sys.modules.copy()
    try:
        yield
    finally:
        sys.modules.clear()
        sys.modules.update(before)


def ensure_stub_package(name: str, path: Path | str | None = None) -> types.ModuleType:
    """Register a dummy package for ``name`` only if it is not already imported.

    Never assigns ``__path__`` on an existing *package*: emptying that list
    would hide real submodules and break later tests in the same process.
    If ``path`` is given and the existing module is not a package, ``__path__``
    is set so submodule imports can resolve.
    """
    path_str = str(path) if path is not None else None
    existing = sys.modules.get(name)
    if existing is not None:
        if path_str is not None and not hasattr(existing, "__path__"):
            existing.__path__ = [path_str]  # type: ignore[attr-defined]
        return existing
    mod = types.ModuleType(name)
    mod.__path__ = [path_str] if path_str is not None else []  # type: ignore[attr-defined]
    sys.modules[name] = mod
    return mod


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
