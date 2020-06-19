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

"""Partitioned PS StrategyBuilder with Greedy Load Balancer."""

from math import ceil
from tensorflow.python.framework import tensor_shape

from autodist.const import ENV
from autodist.kernel.common.op_info import CONTROL_FLOW_OPS
from autodist.kernel.common.utils import get_consumers, get_op_name
from autodist.kernel.partitioner import PartitionerConfig
from autodist.strategy.base import Strategy, StrategyBuilder
from autodist.proto import strategy_pb2


class PartitionedPS(StrategyBuilder):
    """
    Partitioned PS StrategyBuilder with Greedy Load Balancer.

    Determine the number of partitions for each partition-able
    variable by finding its minimum divisor along the first axis.

    Then, use a greedy load balancer (determined by memory usage)
    to assign the partitions to Parameter Servers. This means that,
    unlike the standard PS StrategyBuilder, that a variable can be
    spread across multiple servers using this StrategyBuilder.
    """

    def __init__(self, local_proxy_variable=False, sync=True, staleness=0):
        self._local_proxy_variable = local_proxy_variable
        self._sync = sync
        self._staleness = staleness
        if self._staleness > 0:
            assert self._sync, 'If staleness is positive, sync has to be set True.'
        self.loads = {}

    def build(self, graph_item, resource_spec):
        """Generate the Strategy."""
        expr = Strategy()

        # get each variable, generate variable synchronizer config
        expr.graph_config.replicas.extend([k for k, v in resource_spec.gpu_devices])
        for k, v in resource_spec.node_cpu_devices.items():
            if k not in resource_spec.node_gpu_devices:
                expr.graph_config.replicas.extend(v)

        # find all variables
        variables = graph_item.trainable_var_op_to_var.values()
        reduction_device_names = [k for k, _ in resource_spec.cpu_devices]
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
        if (len(self.loads) < 1 and not ENV.AUTODIST_IS_TESTING.val) or \
                any((o.type in CONTROL_FLOW_OPS for o in get_consumers(var.op))):
            # Don't partition if there is only one reduction device or if the variable is connected to control flow
            # For stability, we err on the side of not partitioning over potentially breaking
            num_shards = 1
        else:
            num_shards = self.get_num_shards(var)

        # Determine placement of vars/parts
        sorted_ps = sorted(self.loads, key=self.loads.get)
        if num_shards > len(self.loads):
            # If there's more shards than servers, round-robin in greedy order
            sorted_ps = sorted_ps * ceil(num_shards / len(self.loads))
        min_ps = sorted_ps[0:num_shards]
        for ps in min_ps:
            self.loads[ps] += byte_size_load_fn(var) / num_shards

        # setup node config
        node = strategy_pb2.Strategy.Node()
        node.var_name = var.name

        if num_shards == 1:
            node.PSSynchronizer.reduction_destination = min_ps[0]
            node.PSSynchronizer.local_replication = self._local_proxy_variable
            node.PSSynchronizer.sync = self._sync
            node.PSSynchronizer.staleness = self._staleness
        else:
            # generate the partitioner config
            shape = var.initial_value.shape
            partition_list = [1] * len(var.initial_value.shape)
            partition_axis = 0
            partition_list[partition_axis] = min(num_shards, shape.dims[partition_axis].value)
            pc = PartitionerConfig(partition_list=partition_list)
            node.partitioner = pc.partition_str

            for i in range(num_shards):
                part = strategy_pb2.Strategy.Node()
                part.var_name = '{}/part_{}:0'.format(get_op_name(var.name), i)
                part.PSSynchronizer.reduction_destination = min_ps[i]
                part.PSSynchronizer.local_replication = self._local_proxy_variable
                part.PSSynchronizer.sync = self._sync
                part.PSSynchronizer.staleness = self._staleness
                node.part_config.extend([part])
        return node

    @staticmethod
    def get_num_shards(var):
        """Gets the minimum number of shards for a variable."""
        if not var.initial_value.shape.ndims:
            return 1

        n = int(var.initial_value.shape[0])
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
