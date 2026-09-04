"""
Multiprocessing Executor for parallel task execution.

Handles multiprocessing using apply_async with support for
exception handling and optional progress bars.
"""

import multiprocessing as mp
from typing import Any

from tqdm import tqdm

from .. import terminal_output


def start_plot_process(
    target,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
) -> mp.Process | None:
    """Run a documentary plot without blocking the parent pipeline.

    Starts a child process when the caller is not a daemon. Pool workers
    (``extract_multiprocessing``) are daemons and cannot spawn children, so
    the plot then runs in the current process.
    """
    kwargs = kwargs or {}
    if mp.current_process().daemon:
        target(*args, **kwargs)
        return None
    process = mp.Process(target=target, args=args, kwargs=kwargs)
    process.start()
    return process


class Executor:
    """
    Class that handles the multiprocessing, using apply_async.
    Allows for easy catch of exceptions.

    ``process_num`` of ``None`` or ``<= 0`` uses half the logical CPUs
    (``cpu_count() // 2``).
    """

    def __init__(self, process_num: int | None, **kwargs: Any) -> None:
        if not mp.get_start_method(allow_none=True):
            mp.set_start_method("spawn")

        if process_num is None or process_num <= 0:
            process_num = int(mp.cpu_count() / 2)

        #   Get max_tasks_per_child parameter
        max_tasks_per_child = kwargs.get("maxtasksperchild", None)
        if max_tasks_per_child is None:
            max_tasks_per_child = 6

        #   Init multiprocessing pool
        self.pool: mp.Pool = mp.Pool(
            process_num,
            maxtasksperchild=max_tasks_per_child,
        )
        #   Init variables
        self.res: list[Any] = []
        self.err: Any = None

        #   Add progress bar if requested
        self.progress_bar: tqdm | None = None
        self.add_progress_bar: bool = kwargs.get("add_progress_bar", False)
        n_tasks: int | None = kwargs.get("n_tasks", None)
        if self.add_progress_bar and n_tasks:
            self.progress_bar = tqdm(total=n_tasks)

    def collect_results(self, result: Any) -> None:
        """
        Uses apply_async's callback to set up a separate Queue
        for each process.
        """
        #   Update progress bar
        if isinstance(self.progress_bar, tqdm):
            self.progress_bar.update(1)

        #   Catch all results
        self.res.append(result)

    def callback_error(self, e: BaseException) -> None:
        """
        Handles exceptions by apply_async's error callback.
        """
        terminal_output.print_to_terminal(
            "Exception detected: Try to terminate the multiprocessing Pool",
            style_name="ERROR",
        )
        terminal_output.print_to_terminal(
            f"The exception is: {e}",
            style_name="ERROR",
        )
        #   Terminate pool
        self.pool.terminate()

        #   Terminate progress bar
        if isinstance(self.progress_bar, tqdm):
            self.progress_bar.close()
        self.progress_bar = None

        #   Raise exceptions
        self.err = e
        raise e

    def schedule(
        self,
        function: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Call to apply_async.
        """
        if kwargs is None:
            kwargs = {}

        self.pool.apply_async(
            function,
            args,
            kwargs,
            callback=self.collect_results,
            error_callback=self.callback_error,
        )

    def wait(self) -> None:
        """
        Close pool and wait for completion.
        """
        try:
            self.pool.close()
            self.pool.join()
        finally:
            #   Terminate progress bar
            if isinstance(self.progress_bar, tqdm):
                self.progress_bar.close()
            self.progress_bar = None

    def __del__(self) -> None:
        if hasattr(self, "pool") and self.pool:
            self.pool.terminate()
            self.pool.join()
