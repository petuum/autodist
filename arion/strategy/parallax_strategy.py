"""Parallax StrategyBuilder."""
from tensorflow.python.framework import ops

from autodist.strategy.base import Strategy
from autodist.strategy.ps_lb_strategy import PSLoadBalancing
from autodist.strategy.all_reduce_strategy import AllReduce
from autodist.kernel.common.utils import get_op_name


class Parallax(PSLoadBalancing, AllReduce):
    """
    Generates the Parallax Strategy from https://arxiv.org/pdf/1808.02621.pdf.

    The Parallax strategy mixes Parameter Server and AllReduce. The rationale is that
    a PS architecture is more suitable for sparse gradient updates, while AllReduce
    has reportedly better performance on dense gradient updates.
    """

    def __init__(self, chunk_size=128, local_proxy_variable=False, sync=True):
        PSLoadBalancing.__init__(self, local_proxy_variable, sync)
        AllReduce.__init__(self, chunk_size)

    # pylint: disable=attribute-defined-outside-init
    def build(self, graph_item, resource_spec):
        """Generate the strategy."""
        expr = Strategy()

        # For each variable, generate variable synchronizer config
        expr.graph_config.replicas.extend([k for k, v in resource_spec.gpu_devices])
        reduction_device_names = [k for k, _ in resource_spec.cpu_devices]
        self.loads = {ps: 0.0 for ps in reduction_device_names}

        # Generate node config
        node_config = []
        for var in graph_item.get_trainable_variables():
            var_op_name = get_op_name(var.name)
            grad, _, _ = graph_item.var_op_name_to_grad_info[var_op_name]
            if isinstance(grad, ops.Tensor):  # this is a dense variable
                config = self._gen_all_reduce_node_config(var.name, 'RING')
            else:  # sparse updates
                # For Parallax Strategy, all PS vars are sparse so we don't use a proxy.
                # Sparse variables are likely larger, so keeping copies would be costlier,
                # and usually each device only requires a small part of the overall variable.
                config = self._gen_ps_node_config(
                    var,
                    False,
                    self._sync
                )
            node_config.append(config)
        expr.node_config.extend(node_config)

        return expr
