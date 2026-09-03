"""Core utilities shared across ost_photometry submodules."""

from .parallel import Executor, start_plot_process

__all__ = ["Executor", "start_plot_process"]
