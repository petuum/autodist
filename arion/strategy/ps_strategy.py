"""PS Strategy."""

from autodist.strategy.base import Strategy, StrategyBuilder
from autodist.item import Item


class PS(StrategyBuilder):
    """PS Strategy."""

    def __init__(self, item: Item, resource_spec):
        super().__init__(item, resource_spec)

    def _build(self):
        expr = Strategy()

        # get each variable, generate variable synchronizer config
        expr.graph_config['replicas'] = {k for k, v in self._resource_spec.gpu_devices}
        # find all variables
        variables = self._item.get_variables_to_sync()
        reduction_device_names = [k for k, _ in self._resource_spec.cpu_devices]
        for var in variables:
            config = self._gen_ps_node_config(reduction_device_names)
            expr.node_config.update({var.name: config})
        return expr

    @staticmethod
    def _gen_ps_node_config(reduction_destinations):
        node_config = {
            'synchronizer': {
                'type': 'ps',
                'config': {
                    'reduction_destinations': reduction_destinations
                }
            }
        }
        return node_config
