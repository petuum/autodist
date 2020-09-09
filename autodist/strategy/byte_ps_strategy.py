"""BytePS StrategyBuilder."""
from autodist.strategy.base import Strategy
from autodist.strategy.ps_lb_strategy import PSLoadBalancing


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
