"""PS Load Balancing Strategy."""

from tensorflow.python.framework import tensor_shape

from autodist.strategy.base import Strategy, StrategyBuilder


class PSLoadBalancing(StrategyBuilder):
    """PS Strategy with Greedy Load Balancing."""

    def __init__(self, *args, **kwargs):
        self.loads = {}
        super().__init__(*args, **kwargs)

    def _build(self):
        expr = Strategy()

        # get each variable, generate variable synchronizer config
        expr.graph_config['replicas'] = {k for k, v in self._resource_spec.gpu_devices}
        # find all variables
        variables = self._item.get_trainable_variables()
        reduction_device_names = [k for k, _ in self._resource_spec.cpu_devices]
        self.loads = {ps: 0.0 for ps in reduction_device_names}

        # Mark each variable to be synchronized with a Parameter Server
        node_config = {var.name: self._gen_ps_node_config(var) for var in variables}
        expr.node_config.update(node_config)

        return expr

    def _gen_ps_node_config(self, var):
        """
        Creates a NodeConfig specifying synchronization with Parameter Servers.

        :param reduction_destinations: The location of the parameter servers.
        :return:
            Dict
        """
        min_ps = min(self.loads, key=self.loads.get)
        self.loads[min_ps] += byte_size_load_fn(var)

        node_config = {
            'synchronizer': {
                'type': 'PSSynchronizer',
                'config': {
                    'reduction_destinations': [min_ps],
                    'local_replication': False,
                    'sync': True
                }
            }
        }
        return node_config


def byte_size_load_fn(op):
    """
    Load function that computes the byte size of a single-output `Operation`.

    Copied (with modifications) from tensorflow.contrib.training.python.training.device_setter.

    This is intended to be used with `"Variable"` ops, which have a single
    `Tensor` output with the contents of the variable.  However, it can also be
    used for calculating the size of any op that has a single output.

    Intended to be used with `GreedyLoadBalancingStrategy`.

    Args:
      op: An `Operation` with a single output, typically a "Variable" op.

    Returns:
      The number of bytes in the output `Tensor`.

    Raises:
      ValueError: if `op` does not have a single output, or if the shape of the
        single output is not fully-defined.
    """
    elem_size = op.dtype.size
    shape = op.get_shape()
    if not shape.is_fully_defined():
        # Due to legacy behavior, scalar "Variable" ops have output Tensors that
        # have unknown shape when the op is created (and hence passed to this
        # load function for placement), even though the scalar shape is set
        # explicitly immediately afterward.
        shape = tensor_shape.TensorShape(op.get_attr("shape"))
    shape.assert_is_fully_defined()
    return shape.num_elements() * elem_size
