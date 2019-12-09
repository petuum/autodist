"""AllReduce strategy."""

from autodist.strategy.base import Strategy, StrategyBuilder
from autodist.proto import strategy_pb2, synchronizers_pb2


class AllReduce(StrategyBuilder):
    """AllReduce Strategy."""

    def build(self, graph_item, resource_spec):
        """Build it."""
        expr = Strategy()

        # get each variable, generate variable synchronizer config
        expr.graph_config.replicas.extend([k for k, v in resource_spec.gpu_devices])
        # find all variables
        variables = graph_item.get_trainable_variables()

        # Mark each variable to be synchronized with allreduce
        node_config = [self._gen_all_reduce_node_config(var.name) for var in variables]
        expr.node_config.extend(node_config)

        return expr

    @staticmethod
    def _gen_all_reduce_node_config(var_name, all_reduce_spec="AUTO", compressor="PowerSGDCompressor"):
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
        return node
