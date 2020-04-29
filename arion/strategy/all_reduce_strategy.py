"""AllReduce StrategyBuilder."""

from autodist.strategy.base import Strategy, StrategyBuilder
from autodist.proto import strategy_pb2, synchronizers_pb2


class AllReduce(StrategyBuilder):
    """
    AllReduce StrategyBuilder.

    This StrategyBuilder generates a strategy that synchronizes every dense variable using AllReduce.
    It will sequentially merge collective ops into a single collective group based on chunk_size.

    This strategy does not support synchronizing sparse updates with >1 nodes due to the TF AllGather bug.
    """

    def __init__(self, chunk_size=128):
        """
        Init function.

        Args:
            chunk_size (int): chunk_size is a positive integer indicating how many variables will be merged
                              sequentially as a group by scoped allocator.
        """
        if chunk_size < 1:
            raise ValueError('The chunk_size must be greater than zero.')
        self.chunk_size = chunk_size 

    def build(self, graph_item, resource_spec):
        """Generate the strategy."""
        expr = Strategy()

        # get each variable, generate variable synchronizer config
        expr.graph_config.replicas.extend([k for k, v in resource_spec.gpu_devices])
        # find all variables
        variables = graph_item.get_trainable_variables()

        # Mark each variable to be synchronized with allreduce
        for i, var in enumerate(variables):
            group_id = i // self.chunk_size
            node_config = self._gen_all_reduce_node_config(var.name, group=group_id)
            expr.node_config.append(node_config)

        return expr

    @staticmethod
    def _gen_all_reduce_node_config(var_name, group=0, all_reduce_spec="AUTO", compressor="PowerSGDCompressor"):
        """
        Creates a NodeConfig specifying synchronization with AllReduce.

        Args:
            var_name (str): The name of the variable.
            group (int): the collective group this synchronizer belongs to.
            algo (str): 'AUTO', 'NCCL', 'RING'.
            compressor (str): Gradient compression algorithm to use.

        Returns:
            strategy_pb2.Strategy.Node: the config for the node.
        """
        node = strategy_pb2.Strategy.Node()
        node.var_name = var_name
        node.AllReduceSynchronizer.spec = synchronizers_pb2.AllReduceSynchronizer.Spec.Value(all_reduce_spec)
        node.AllReduceSynchronizer.compressor = synchronizers_pb2.AllReduceSynchronizer.Compressor.Value(compressor)
        node.AllReduceSynchronizer.group = group
        return node
