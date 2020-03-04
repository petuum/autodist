"""PS StrategyBuilder."""

from autodist.strategy.base import Strategy, StrategyBuilder
from autodist.proto import strategy_pb2


class PS(StrategyBuilder):
    """
    PS StrategyBuilder.

    Generates a Strategy that synchronizes every variable
    using Parameter Servers. Each variable is only assigned
    to one Parameter Server.
    """

    def __init__(self, local_proxy_variable=False, sync=True, staleness=0):
        self._local_proxy_variable = local_proxy_variable
        self._sync = sync
        self._staleness = staleness
        if self._staleness > 0:
            assert self._sync, 'If staleness is positive, sync has to be set true.'

    def build(self, graph_item, resource_spec):
        """Build PS strategy."""
        expr = Strategy()

        # get each variable, generate variable synchronizer config
        expr.graph_config.replicas.extend([k for k, v in resource_spec.gpu_devices])
        # find all variables
        variables = graph_item.get_trainable_variables()
        reduction_device_names = [k for k, _ in resource_spec.cpu_devices][0:1]

        # Mark each variable to be synchronized with a Parameter Server
        node_config = [self._gen_ps_node_config(var.name, reduction_device_names, self._local_proxy_variable,
                                                self._sync, self._staleness)
                       for var in variables]
        expr.node_config.extend(node_config)
        return expr

    @staticmethod
    def _gen_ps_node_config(var_name, reduction_destinations, local_proxy_variable, sync, staleness):
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
        node.PSSynchronizer.local_replication = local_proxy_variable
        node.PSSynchronizer.sync = sync
        node.PSSynchronizer.staleness = staleness
        return node
