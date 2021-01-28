"""BytePS StrategyBuilder(s)."""
from autodist.strategy.base import Strategy, StrategyBuilder
from autodist.proto import strategy_pb2
from autodist.strategy.ps_lb_strategy import PSLoadBalancing, byte_size_load_fn


class BytePS(PSLoadBalancing):
    """
    Generates the BytePS Strategy from https://github.com/bytedance/byteps.

    The BytePS strategy exploits CPU-only nodes for communication while GPU nodes
    for computatoin.
    """

    def __init__(self, local_proxy_variable=False, sync=True, staleness=0):
        PSLoadBalancing.__init__(self, local_proxy_variable, sync, staleness)

    # pylint: disable=attribute-defined-outside-init
    def build(self, graph_item, resource_spec):
        """Generate the strategy."""
        expr = Strategy()

        # get each variable, generate variable synchronizer config
        expr.graph_config.replicas.extend([k for k, v in resource_spec.gpu_devices])

        # find all variables
        variables = graph_item.get_trainable_variables()
        reduction_device_names = [k for k, _ in resource_spec.cpu_only_devices]
        self.loads = {ps: 0.0 for ps in reduction_device_names}

        # Mark each variable to be synchronized with a Parameter Server
        node_config = [self._gen_ps_node_config(var, self._local_proxy_variable, self._sync, self._staleness)
                       for var in variables]
        expr.node_config.extend(node_config)

        return expr


class MultinomialBytePS(StrategyBuilder):
    """
    BytePS with Multinomial Load Balancing.

    Each PS gets assigned variables of which size is proportional to
    its bandwidth.
    """

    def __init__(self, local_proxy_variable=False, sync=True, staleness=0):
        self._local_proxy_variable = local_proxy_variable
        self._sync = sync
        self._staleness = staleness
        if self._staleness > 0:
            assert self._sync, 'If staleness is positive, sync has to be set true'
        self.loads = {}
        self.bandwidth = {}
        super().__init__()

    def build(self, graph_item, resource_spec):
        """Generate the Strategy."""
        expr = Strategy()

        # get each variable, generate variable synchronizer config
        expr.graph_config.replicas.extend([k for k, v in resource_spec.gpu_devices])
        for k, v in resource_spec.node_cpu_devices.items():
            if k not in resource_spec.node_gpu_devices:
                expr.graph_config.replicas.extend(v)

        # find all variables
        variables = graph_item.get_trainable_variables()
        reduction_device_names = [k for k, _ in resource_spec.cpu_devices]
        bandwidth = resource_spec.network_bandwidth
        self.bandwidth = {ps: bandwidth[ps.split(':')[0]] for ps in reduction_device_names}
        self.loads = {ps: 1e-8 / self.bandwidth[ps] for ps in reduction_device_names}

        # Mark each variable to be synchronized with a Parameter Server
        node_config = [self._gen_ps_node_config(var, self._local_proxy_variable, self._sync, self._staleness)
                       for var in variables]
        expr.node_config.extend(node_config)

        return expr

    def _gen_ps_node_config(self, var, local_proxy_variable, sync, staleness):
        """
        Creates a NodeConfig specifying synchronization with Parameter Servers.

        Args:
            var (Variable): The variable to generate a config for.

        Returns:
            strategy_pb2.Strategy.Node: the config for the node.
        """
        min_ps = min(self.loads, key=self.loads.get)
        self.loads[min_ps] += byte_size_load_fn(var) / self.bandwidth[min_ps]

        node = strategy_pb2.Strategy.Node()
        node.var_name = var.name
        node.PSSynchronizer.reduction_destination = min_ps
        node.PSSynchronizer.local_replication = local_proxy_variable
        node.PSSynchronizer.sync = sync
        node.PSSynchronizer.staleness = staleness
        return node
