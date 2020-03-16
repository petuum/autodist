"""AllReduce StrategyBuilder."""

from autodist.strategy.base import Strategy, StrategyBuilder
from autodist.proto import strategy_pb2, synchronizers_pb2


class AllReduce(StrategyBuilder):
    """
    AllReduce StrategyBuilder.

    This StrategyBuilder generates a strategy that
    synchronizes every dense variable using AllReduce while every sparse var using .
    """

    def __init__(self, chunk_size=128):
        """
        Init function.

        Args:
            chunk_size (int): chunk_size is a positive integer
                              used by scoped allocator.
        """
        self.chunk_size = chunk_size 

    def build(self, graph_item, resource_spec):
        """Generate the strategy."""
        expr = Strategy()

        # get each variable, generate variable synchronizer config
        expr.graph_config.replicas.extend([k for k, v in resource_spec.gpu_devices])
        # find all variables
        variables = graph_item.get_trainable_variables()

        # Mark each variable to be synchronized with allreduce
        node_config = [self._gen_all_reduce_node_config(var.name) for var in variables]
        expr.node_config.extend(node_config)

        return expr

    def _gen_all_reduce_node_config(self, var_name, all_reduce_spec="AUTO", compressor="PowerSGDCompressor"):
        """
        Creates a NodeConfig specifying synchronization with AllReduce.

        Args:
            var_name (str): The name of the variable.
            algo (str): 'AUTO', 'NCCL', 'RING'.
            compressor (str): Gradient compression algorithm to use.
            TODO(Hao): add more specs and descriptions for each allreduce spec.

        Returns:
            strategy_pb2.Strategy.Node: the config for the node.
        """
        node = strategy_pb2.Strategy.Node()
        node.var_name = var_name
        node.AllReduceSynchronizer.spec = synchronizers_pb2.AllReduceSynchronizer.Spec.Value(all_reduce_spec)
        node.AllReduceSynchronizer.compressor = synchronizers_pb2.AllReduceSynchronizer.Compressor.Value(compressor)
        node.AllReduceSynchronizer.chunk_size = self.chunk_size
        return node
