"""Parallax strategy."""
from tensorflow.python.framework import ops

from autodist.strategy.base import Strategy
from autodist.strategy.ps_lb_strategy import PSLoadBalancing
from autodist.strategy.all_reduce_strategy import AllReduce
from autodist.kernel.common.utils import get_op_name


class Parallax(PSLoadBalancing, AllReduce):
    """
    Parallax Strategy from https://arxiv.org/pdf/1808.02621.pdf.

    Parallax strategy mixes parameter server and allreduce. The rationale is that
    a ps architecture is more suitable for sparse gradient updates, while allreduce
    has reportedly better performance on dense gradient updates.
    """

    def __init__(self, chunk_size=128, local_proxy_variable=False, sync=True):
        PSLoadBalancing.__init__(self, local_proxy_variable, sync)
        AllReduce.__init__(self, chunk_size)

    # pylint: disable=attribute-defined-outside-init
    def build(self, graph_item, resource_spec):
        """Build it."""
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
                config = self._gen_ps_node_config(
                    var,
                    False,  # For Parallax Strategy, all PS vars are sparse which does not need proxy.
                    self._sync
                )
            node_config.append(config)
        expr.node_config.extend(node_config)

        return expr