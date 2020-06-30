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

"""Synchronizer."""
from abc import ABC, abstractmethod
from tensorflow.python import ops

from autodist.kernel.common.utils import get_op_name, update_consumers, update_control_consumers, replica_prefix, \
    strip_replica_prefix, get_index_from_tensor_name


class Synchronizer(ABC):
    """
    Synchronizer.

    Given a variable, can modify the TF Graph to synchronize its
    gradients in either an in-graph or a between-graph fashion.

    - In-graph means the synchronization happens in one `tf.Graph`
    - Between-graph means the synchronization happens across
        multiple `tf.Graphs` (e.g., each worker has its own graph)
    """

    def __init__(self):
        self.num_workers = None
        self.num_replicas = None
        self.worker_device = None
        self.worker_id = None
        self.var_op_to_agg_grad = None
        self.var_op_to_accum_apply_op = None
        self.is_chief = None
        self.all_canonical_replica_devices = None

    # pylint: disable=too-many-arguments
    def assign_cluster_information(self,
                                   num_workers,
                                   num_replicas,
                                   worker_device,
                                   worker_id,
                                   canonical_replica_devices,
                                   is_chief=False):
        """Store cluster information in the synchronizer."""
        self.num_workers = num_workers
        self.num_replicas = num_replicas
        self.worker_device = worker_device  # local worker device
        self.worker_id = worker_id  # local worker id
        self.all_canonical_replica_devices = canonical_replica_devices
        self.is_chief = is_chief
        return self

    @abstractmethod
    def in_graph_apply(self, graph_item, var_name):
        """
        Apply in-graph synchronization to the grad and target in the graph.

        Args:
            graph_item (graph_item.GraphItem): The graph to put the new ops in.
            var_name (str): The variable name w/o the replica prefix.

        Returns:
            graph_item.GraphItem
        """
        return

    @abstractmethod
    def between_graph_apply(self, graph_item, var_name):
        """
        Apply between-graph synchronization to the target ops in the graph.

        Args:
            graph_item (graph_item.GraphItem): The graph to put the new ops in.
            var_name (str): The variable name w/o the replica prefix.

        Returns:
            graph_item.GraphItem
        """
        return

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
        old_op_name = strip_replica_prefix(get_op_name(old_tensor_name))
        replica_0_op_name = ops.prepend_name_scope(old_op_name, replica_prefix(0))
        replica_0_op = new_graph_item.graph.get_operation_by_name(replica_0_op_name)
        output_idx = get_index_from_tensor_name(old_tensor_name)
        replica_0_tensor = replica_0_op.outputs[output_idx]

        update_consumers(consumer_ops, replica_0_tensor, new_tensor)
        update_control_consumers(control_consumer_ops, replica_0_tensor.op, new_tensor.op)
