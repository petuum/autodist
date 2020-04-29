"""Partitioned PS StrategyBuilder with Greedy Load Balancer."""

import numpy as np

from autodist.kernel.common.utils import get_op_name
from autodist.kernel.partitioner import PartitionerConfig
from autodist.proto import strategy_pb2, synchronizers_pb2
from autodist.strategy.base import Strategy, StrategyBuilder


class PartitionedAR(StrategyBuilder):
    """
    Partitioned AR StrategyBuilder.

    This StrategyBuilder generates a strategy that partitions each variable along its first dimension,
    and synchronizes them using AllReduce. It might be advantageous for communicating extremely large
    messages -- when synchronizing a single message is bounded by single-flow bandwidth.
    This strategy does not support synchronizing sparse updates with >1 nodes due to the TF AllGather bug.
    """

    def __init__(self, chunk_size=128):
        self.chunk_size = chunk_size

    def build(self, graph_item, resource_spec):
        """Generate the Strategy."""
        expr = Strategy()

        # data-parallel graph replication first
        expr.graph_config.replicas.extend([k for k, v in resource_spec.gpu_devices])
        # find all variables
        variables = graph_item.trainable_var_op_to_var.values()

        # Mark each variable to be synchronized with allreduce
        node_config = [self._gen_node_config(var) for var in variables]
        expr.node_config.extend(node_config)

        return expr

    def _gen_node_config(self, var):
        """
        Creates a NodeConfig specifying partitioning and synchronization with AllReduce.

        Args:
            var (Variable): The variable to generate a config for.

        Returns:
            Dict: the config dict for the node.
        """
        num_shards = self.get_num_shards(var)

        node = strategy_pb2.Strategy.Node()
        node.var_name = var.name

        if num_shards <= 1:
            node.AllReduceSynchronizer.spec = synchronizers_pb2.AllReduceSynchronizer.Spec.Value("AUTO")
            node.AllReduceSynchronizer.compressor = \
                synchronizers_pb2.AllReduceSynchronizer.Compressor.Value("PowerSGDCompressor")
            node.AllReduceSynchronizer.chunk_size = self.chunk_size
            return node

        # num_parts > 1 means the variable will be partitioned
        # generate the partitioner config
        shape = var.initial_value.shape
        partition_list = [1] * len(var.initial_value.shape)
        partition_axis = 0
        partition_list[partition_axis] = min(num_shards, shape.dims[partition_axis].value)
        num_parts = np.prod(partition_list)
        pc = PartitionerConfig(partition_list=partition_list)
        node.partitioner = pc.partition_str
        for i in range(num_parts):
            part = strategy_pb2.Strategy.Node()

            # If part var_name is inconsistent with what TF will create, partitioner kernel will correct it later.
            # Here let's just make it consistent
            part.var_name = '{}/part_{}:0'.format(get_op_name(var.name), i)
            part.AllReduceSynchronizer.spec = synchronizers_pb2.AllReduceSynchronizer.Spec.Value("AUTO")
            part.AllReduceSynchronizer.compressor = \
                synchronizers_pb2.AllReduceSynchronizer.Compressor.Value("PowerSGDCompressor")
            part.AllReduceSynchronizer.chunk_size = self.chunk_size
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
