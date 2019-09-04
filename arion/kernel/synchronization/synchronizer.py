"""Synchronizer."""
from abc import ABC, abstractmethod
from tensorflow.python import ops

from autodist.kernel.common.utils import get_op_name, update_consumers, update_control_consumers, replica_prefix


class Synchronizer(ABC):
    """Synchronizer."""

    # a static context to record the load balancing status
    # and make some adjustment when necessary
    context = {}

    def __init__(self):
        self.num_workers = None
        self.num_replicas = None
        self.worker_device = None
        self.worker_id = None
        self.var_op_to_agg_grad = None
        self.var_op_to_accum_apply_op = None

    def assign_cluster_information(self, num_workers, num_replicas, worker_device, worker_id):
        """Store cluster information in the synchronizer."""
        self.num_workers = num_workers
        self.num_replicas = num_replicas
        self.worker_device = worker_device
        self.worker_id = worker_id
        return self

    @abstractmethod
    def in_graph_apply(self, graph_item, grad, target):
        """
        Apply in-graph synchronization to the grad and target in the graph.

        Args:
            graph_item (GraphItem): The graph to put the new ops in.
            grad: The gradient object.
            target: The target tensor.

        Returns:
            GraphItem
        """
        pass

    @abstractmethod
    def between_graph_apply(self, *args, **kwargs):
        """
        Apply between-graph synchronization to the target ops in the graph.

        Args:
            *args: Leaving these up to the implementation until we find a universal signature.
            **kwargs: See above.

        Returns:
            GraphItem
        """
        pass

    @classmethod
    def create(cls, name, *args, **kwargs):
        """
        Create new Synchronizer instance given subclass name.

        Args:
            name: Name of the Synchronizer subclass (e.g. PSSynchronizer).
            *args: Any args for the subclass constructor.
            **kwargs: Any kwargs for the subclass constructor.

        Returns:
            Synchronizer
        """
        subclass = next(subclass for subclass in cls.__subclasses__() if subclass.__name__ == name)
        return subclass(*args, **kwargs)

    @staticmethod
    def _update_gradient_consumers(new_graph_item, consumer_ops, control_consumer_ops,
                                   old_tensor_name, new_tensor):
        """Make gradient's consumers consume the aggregated gradient instead of the original one of replica_0."""
        # Get the original tensor (the one from replica 0) to replace
        old_op_name = get_op_name(old_tensor_name)
        replica_0_op_name = ops.prepend_name_scope(old_op_name, replica_prefix(0))
        replica_0_op = new_graph_item.graph.get_operation_by_name(replica_0_op_name)
        output_idx = int(old_tensor_name.split(':')[1])
        replica_0_tensor = replica_0_op.outputs[output_idx]

        update_consumers(consumer_ops, replica_0_tensor, new_tensor)
        update_control_consumers(control_consumer_ops, replica_0_tensor.op, new_tensor.op)
