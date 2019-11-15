"""AllReduce strategy."""

from autodist.strategy.base import Strategy, StrategyBuilder


class AllReduce(StrategyBuilder):
    """AllReduce Strategy."""

    def _build(self):
        expr = Strategy()

        # get each variable, generate variable synchronizer config
        expr.graph_config['replicas'] = {k for k, v in self._resource_spec.gpu_devices}
        # find all variables
        variables = self._item.get_trainable_variables()
        # reduction_device_names = [k for k, _ in self._resource_spec.cpu_devices][0:1]

        # Mark each variable to be synchronized with allreduce
        node_config = {var.name: self._gen_all_reduce_node_config() for var in variables}
        expr.node_config.update(node_config)

        return expr

    @staticmethod
    def _gen_all_reduce_node_config(all_reduce_spec="auto"):
        """
        Creates a NodeConfig specifying synchronization with AllReduce.

        Args:
            algo (str): 'auto', 'nccl', 'ring'.
            TODO(Hao): add more specs and descriptions for each allreduce spec.

        Returns:
            dict: the config dict for the node.

        """
        node_config = {
            'synchronizer': {
                'type': 'AllReduceSynchronizer',
                'config': {
                    'spec': all_reduce_spec,
                    'compressor': 'NoneCompressor',
                }
            }
        }
        return node_config
