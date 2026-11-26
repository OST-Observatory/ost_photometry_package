"""Background plot helper (inline fallback inside daemon pool workers) and
the worker pools of the Executor."""

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


def _noop() -> None:
    pass


def test_pool_spawns_even_when_fork_is_the_global_start_method():
    """A plot process started before the first pool fixes the global start
    method to "fork" on Linux; pool workers forked from a process with
    threads have hung forever in Pool.join. The pool keeps "spawn"."""
    import math
    import multiprocessing as mp

    from ost_photometry.core.parallel import Executor

    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    process = ctx.Process(target=_noop)
    process.start()
    process.join()

    executor = Executor(2, maxtasksperchild=1)  # a new worker for every task
    assert executor.pool._ctx.get_start_method() == "spawn"
    for value in (1.0, 4.0, 9.0, 16.0, 25.0):
        executor.schedule(math.sqrt, args=(value,))
    executor.wait()
    assert executor.err is None and sorted(executor.res) == [1.0, 2.0, 3.0, 4.0, 5.0]
