"""Auto Strategy."""
from .base import StrategyBuilder, Strategy


class Auto(StrategyBuilder):
    """Auto Strategy."""

    def _build(self):
        return Strategy()
