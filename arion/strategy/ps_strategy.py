"""PS Strategy."""

from autodist.strategy.base import Strategy, StrategyBuilder


class PS(StrategyBuilder):
    """PS Strategy."""

    def _build(self):
        expr = Strategy()

        # get each variable, generate variable synchronizer config
        expr.graph_config['replicas'] = {k for k, v in self._resource_spec.gpu_devices}
        # find all variables
        variables = self._item.get_variables_to_sync()
        reduction_device_names = [k for k, _ in self._resource_spec.cpu_devices][0:1]

        # Mark each variable to be synchronized with a Parameter Server
        node_config = {var.name: self._gen_ps_node_config(reduction_device_names) for var in variables}
        expr.node_config.update(node_config)

        return expr

    @staticmethod
    def _gen_ps_node_config(reduction_destinations):
        """
        Creates a NodeConfig specifying synchronization with Parameter Servers.

        :param reduction_destinations: The location of the parameter servers.
        :return:
            Dict
        """
        node_config = {
            'synchronizer': {
                'type': 'PSSynchronizer',
                'config': {
                    'reduction_destinations': reduction_destinations,
                    'local_replication': True,
                    'sync': True
                }
            }
        }
        return node_config
