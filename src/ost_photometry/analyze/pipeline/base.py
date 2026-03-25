"""Base class for pipeline steps."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import PipelineConfig
    from .context import AnalysisContext


class PipelineStep(ABC):
    """Abstract base class for pipeline steps."""

    name: str = ""

    @abstractmethod
    def run(
        self,
        context: "AnalysisContext",
        config: "PipelineConfig",
    ) -> "AnalysisContext":
        """Execute the step. Modify context and return it."""
        pass

    def skip(
        self,
        context: "AnalysisContext",
        config: "PipelineConfig",
    ) -> bool:
        """Return True to skip this step."""
        return False
