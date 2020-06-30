# Copyright 2020 Petuum. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PS Load Balancing StrategyBuilder."""

from tensorflow.python.framework import tensor_shape

from autodist.strategy.base import Strategy, StrategyBuilder
from autodist.proto import strategy_pb2


class PSLoadBalancing(StrategyBuilder):
    """
    PS StrategyBuilder with Greedy Load Balancing.

    The Load Balancing is determined by total memory
    usage for storing variables, i.e. we always assign
    a variable to the current lowest-memory-usage
    Parameter Server.
    """

    def __init__(self, local_proxy_variable=False, sync=True, staleness=0):
        self._local_proxy_variable = local_proxy_variable
        self._sync = sync
        self._staleness = staleness
        if self._staleness > 0:
            assert self._sync, 'If staleness is positive, sync has to be set true.'
        self.loads = {}
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
        self.loads = {ps: 0.0 for ps in reduction_device_names}

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
        self.loads[min_ps] += byte_size_load_fn(var)

        node = strategy_pb2.Strategy.Node()
        node.var_name = var.name
        node.PSSynchronizer.reduction_destination = min_ps
        node.PSSynchronizer.local_replication = local_proxy_variable
        node.PSSynchronizer.sync = sync
        node.PSSynchronizer.staleness = staleness
        return node


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
