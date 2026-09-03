"""Background plot helper: inline fallback inside daemon pool workers."""

from __future__ import annotations

import sys
from types import SimpleNamespace

from helpers import pkg_src

_PKG_SRC = pkg_src()
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))


def test_start_plot_process_runs_inline_when_daemon():
    from unittest.mock import patch

    from ost_photometry.core import parallel as par

    called: list[int] = []

    def target(value):
        called.append(value)

    with (
        patch.object(
            par.mp,
            "current_process",
            return_value=SimpleNamespace(daemon=True),
        ),
        patch.object(par.mp, "Process") as proc_cls,
    ):
        result = par.start_plot_process(target, (7,))
    assert result is None
    assert called == [7]
    proc_cls.assert_not_called()


def test_start_plot_process_starts_child_when_not_daemon():
    from unittest.mock import MagicMock, patch

    from ost_photometry.core import parallel as par

    child = MagicMock()
    with (
        patch.object(
            par.mp,
            "current_process",
            return_value=SimpleNamespace(daemon=False),
        ),
        patch.object(par.mp, "Process", return_value=child) as proc_cls,
    ):
        result = par.start_plot_process(lambda: None, (), {"x": 1})
    proc_cls.assert_called_once()
    child.start.assert_called_once()
    assert result is child
