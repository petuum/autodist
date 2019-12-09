"""Partitioned PS Strategy with Greedy Load Balancer, with the local replication turned on."""

from math import ceil
from tensorflow.python.framework import tensor_shape

from autodist.strategy.base import Strategy, StrategyBuilder
from autodist.proto import strategy_pb2


class PartitionedPSProxy(StrategyBuilder):
    """Partitioned PS Strategy with Greedy Load Balancer."""

    def __init__(self, *args, **kwargs):
        self.loads = {}
        super().__init__(*args, **kwargs)

    def _build(self):
        expr = Strategy()

        # get each variable, generate variable synchronizer config
        expr.graph_config.replicas.extend([k for k, v in self._resource_spec.gpu_devices])
        # find all variables
        variables = self._item.get_trainable_variables()
        reduction_device_names = [k for k, _ in self._resource_spec.cpu_devices]
        self.loads = {ps: 0.0 for ps in reduction_device_names}

        # Mark each variable to be synchronized with a Parameter Server
        node_config = [self._gen_ps_node_config(var) for var in variables]
        expr.node_config.extend(node_config)

        return expr

    def _gen_ps_node_config(self, var):
        """
        Creates a NodeConfig specifying synchronization with Parameter Servers.

        Args:
            var (Variable): The variable to generate a config for.

        Returns:
            Dict: the config dict for the node.
        """
        num_shards = self.get_num_shards(var)
        sorted_ps = sorted(self.loads, key=self.loads.get)
        if num_shards > len(self.loads):
            # If there's more shards than servers, round-robin in greedy order
            sorted_ps = sorted_ps * ceil(num_shards / len(self.loads))
        min_ps = sorted_ps[0:num_shards]
        for ps in min_ps:
            self.loads[ps] += byte_size_load_fn(var) / num_shards

        node = strategy_pb2.Strategy.Node()
        node.var_name = var.name
        node.PSSynchronizer.reduction_destinations.extend(min_ps)
        node.PSSynchronizer.local_replication = True
        node.PSSynchronizer.sync = True
        return node

    def get_num_shards(self, tensor):
        """Gets the minimum number of shards for a variable."""
        var = self._item.trainable_var_op_to_var[tensor.op]
        if not var.initial_value.shape.ndims:
            return 1

        n = var.initial_value.shape[0]
        for i in range(2, n):
            if n % i == 0:
                return i
        return n


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