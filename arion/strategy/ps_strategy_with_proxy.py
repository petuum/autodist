"""PS Strategy, with the local replication turned on."""

from autodist.strategy.base import Strategy, StrategyBuilder
from autodist.proto import strategy_pb2


class PSProxy(StrategyBuilder):
    """PS Strategy."""

    def _build(self):
        expr = Strategy()

        # get each variable, generate variable synchronizer config
        expr.graph_config.replicas.extend([k for k, v in self._resource_spec.gpu_devices])
        # find all variables
        variables = self._item.get_trainable_variables()
        reduction_device_names = [k for k, _ in self._resource_spec.cpu_devices][0:1]

        # Mark each variable to be synchronized with a Parameter Server
        node_config = [self._gen_ps_node_config(var.name, reduction_device_names) for var in variables]
        expr.node_config.extend(node_config)

        return expr

    @staticmethod
    def _gen_ps_node_config(var_name, reduction_destinations):
        """
        Creates a NodeConfig specifying synchronization with Parameter Servers.

        Args:
            var_name (str): The name of the variable.
            reduction_destinations (Iter[str]): The location of the parameter servers.

        Returns:
            strategy_pb2.Strategy.Node: the config for the node.
        """
        node = strategy_pb2.Strategy.Node()
        node.var_name = var_name
        node.PSSynchronizer.reduction_destinations.extend(reduction_destinations)
        node.PSSynchronizer.local_replication = True
        node.PSSynchronizer.sync = True
        return node
