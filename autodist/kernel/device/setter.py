"""Replica Device Setter."""
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import device_spec

from autodist.kernel.common import utils
from autodist.kernel.common.op_info import MUTABLE_STATE_OPS


# class Placement:
#     pass
#
# class GreedyLoadBalancing(Placement):
#     def __init__(self, num_tasks):
#         """Create a new `LoadBalancingStrategy`.
#         This only consider trainable variables on certain tasks.
#         Args:
#             num_tasks: Number of tasks to cycle among.
#         """
#         self._num_tasks = num_tasks
#         self._loads = np.zeros(num_tasks)
#
#     def __call__(self, op):
#         """Choose a task index for the given `Operation`.
#         Args:
#             op: A `Operation` to be placed on certain tasks.
#         Returns:
#             The next task index to use for the `Operation`. Greedily
#             places the op on the least-loaded task so far, as determined
#             by the load function.
#         """
#         task = np.argmin(self._loads)
#         self._loads[task] += byte_size_load_fn(op)
#         return task
#
# class RoundRobin(Placement):
#   """Returns the next task index for placement in round-robin order.
#
#   This class is not to be used directly by users.  See instead
#   `replica_device_setter()` below.
#   """
#
#   def __init__(self, num_tasks):
#     """Create a new `_RoundRobinStrategy`.
#
#     Args:
#       num_tasks: Number of tasks to cycle among.
#     """
#     self._num_tasks = num_tasks
#     self._next_task = 0
#
#   def __call__(self, unused_op):
#     """Choose a task index for the given `Operation`.
#
#     Args:
#       unused_op: An `Operation` to be placed on.
#
#     Returns:
#       The next task index to use for the `Operation`. Returns the next
#       index, in the range `[offset, offset + num_tasks)`.
#     """
#     task = self._next_task
#     self._next_task = (self._next_task + 1) % self._num_tasks
#     return task


class DeviceSetter:
    """Abstract device setter type."""

    pass


class ReplicaDeviceSetter(DeviceSetter):
    """Class to choose devices for Ops in a replicated training setup."""

    def __init__(self, worker_device, synchronizers=None):
        """
        Constructor.

        Args:
          worker_device : replica worker device
          # TODO: unify device setting
        """
        self._state_device_mapping = None
        if synchronizers:
            self._state_device_mapping = {
                utils.get_op_name(tensor_name): s.target_device for tensor_name, s in synchronizers.items()
            }
        self._worker_device = worker_device

    def __call__(self, op):
        """
        Choose a device for `op`.

        Args:
          op: an `Operation`.

        Returns:
          The device to use for the `Operation`.
        """
        current_device = device_spec.DeviceSpecV2.from_string(op.device or "")

        node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
        if self._state_device_mapping and node_def.op in MUTABLE_STATE_OPS:
            # TODO: Variables in the LOCAL_VARIABLES collection should not be
            # placed in the parameter server.
            ps_device = device_spec.DeviceSpecV2.from_string(self._state_device_mapping.get(op.name))

            # current_job, ps_job = current_device.job, ps_device.job
            # if ps_job and (not current_job or current_job == ps_job):
            #   ps_device = ps_device.replace(task=ps)

            ps_device = ps_device.make_merged_spec(current_device)
            return ps_device.to_string()

        worker_device = device_spec.DeviceSpecV2.from_string(self._worker_device or "")
        worker_device = worker_device.make_merged_spec(current_device)
        return worker_device.to_string()
