"""PS Strategy."""

from autodist.strategy.base import Strategy, StrategyBuilder
from autodist.item import Item


class PS(StrategyBuilder):
    """PS Strategy."""

    def __init__(self, item: Item, resource_spec):
        super().__init__(item, resource_spec)

    def _build(self):
        expr = Strategy()
        expr.graph_config['num_replica'] = self._resource_spec.num_gpus()

        # find all variables
        variables = self._item.get_variables()
        for node in variables:
            node_config = {
                'P': 4,
                'synchronizer': {
                    'type': 'ps',
                    'config': {
                        'reduction_devices': self._resource_spec.get_cpu_devices(),
                        'num_partition': 4
                    }
                }
            }
            expr.node_config[node.name] = node_config
        return expr
