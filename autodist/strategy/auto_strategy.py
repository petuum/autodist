"""Auto Strategy."""
from autodist.strategy.base import StrategyBuilder, Strategy


class Auto(StrategyBuilder):
    """Auto Strategy."""

    def _build(self):
        print(self)
        return Strategy()
