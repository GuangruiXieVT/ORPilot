"""Abstract base class for OR solvers."""

from abc import ABC, abstractmethod

from aifor.models.solution import SolutionResult


class BaseSolver(ABC):
    """Unified interface for OR solvers."""

    name: str = "base"
    framework: str = "base"

    @abstractmethod
    def solve(self, code: str, data: dict) -> SolutionResult:
        """Execute solver code with the given data and return results.

        Args:
            code: Python source code containing a `solve(data) -> dict` function.
            data: Data dictionary to pass to the solve function.

        Returns:
            SolutionResult with status, objective value, variables, etc.
        """
        ...
