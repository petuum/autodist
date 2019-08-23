"""Common helpers for resource variables."""

from collections import defaultdict

from tensorflow.python import ops
from tensorflow.python.ops import variable_scope, state_ops
from tensorflow.python.framework import device_spec
from tensorflow.python.ops import gen_control_flow_ops

from autodist.const import InitOps
from autodist.kernel.common.utils import get_consumers, update_consumers, replica_prefix, AUTODIST_REPLICA_PREFIX


def get_read_var_ops(var_handle_op):
    """Given a resource handle op, get all its read variable ops."""
    return {consumer for consumer in get_consumers(var_handle_op) if consumer.type == "ReadVariableOp"}


def get_read_var_tensor(var_handle_op):
    """Given a resource handle, get the tensor of its default readable value."""
    for read_var_op in get_read_var_ops(var_handle_op):
        if read_var_op.name.endswith("/Read/ReadVariableOp"):
            return read_var_op.outputs[0]
    return None
    # # another way
    # return ops.get_default_graph().get_tensor_by_name(var_handle.op.name + "/Read/ReadVariableOp:0")


def gen_mirror_var_init_op(variable_replicators):
    """Given a list of variable replicators, generate an "initialize" op to trigger the AssignOps of all."""
    all_mirror_var_init_ops = sum([m.mirror_var_init_ops for m in variable_replicators if m], [])
    if all_mirror_var_init_ops:
        with ops.control_dependencies(all_mirror_var_init_ops):
            gen_control_flow_ops.no_op(name=InitOps.MIRROR_VARIABLE_INIT_OP.value)


class ResourceVariableReplicator:
    """Resource Variable Replicator."""

    def __init__(self, resource_var):
        self._this = resource_var
        self._read_var_ops = get_read_var_ops(resource_var.op)
        self._consumer_to_read_var_op = {c: o for o in self._read_var_ops for c in get_consumers(o)}
        self._read_var_op_to_consumers = {o: get_consumers(o) for o in self._read_var_ops}
        self._mirror_vars = []
        self.mirror_var_init_ops = []
        self._read_var_ops_mappings = defaultdict(dict)

    def build_mirror_vars(self, mirror_var_device, num_replicas):
        """Build mirror vars for the master var."""
        master_var = self._this
        original_var_device = device_spec.DeviceSpecV2.from_string(master_var.op.device)
        mv_mapping = self._mirror_vars
        init_ops = self.mirror_var_init_ops

        is_gpu = original_var_device.device_type.upper() == 'GPU'
        if not is_gpu:
            num_replicas = 1
        for i in range(num_replicas):
            if is_gpu:
                mirror_var_device = mirror_var_device.replace(device_type='GPU', device_index=i)
            else:
                mirror_var_device = mirror_var_device.replace(device_type='CPU', device_index=0)
            with ops.device(mirror_var_device):
                prefix = replica_prefix(i) if is_gpu else replica_prefix('CPU')
                mirror_var = variable_scope.get_variable(
                    ops.prepend_name_scope(master_var.op.name, prefix),
                    dtype=master_var.dtype.base_dtype,
                    initializer=master_var.initial_value,
                    trainable=False,
                    collections=[ops.GraphKeys.LOCAL_VARIABLES]
                )
            mv_mapping.append(mirror_var)
            init_ops.append(state_ops.assign(mirror_var, master_var))

        self._mirror_all_read_var_ops()
        self._update_all_consumers()
        return self

    def get_all_update_ops(self, grad_apply_finished, worker_device=None):
        """Create and return new update ops for mirror vars."""
        with ops.device(worker_device):
            with ops.control_dependencies(grad_apply_finished):
                updated_value = self._this.read_value()
        update_ops = []
        for mirror_var in self._mirror_vars:
            with ops.device(mirror_var.device):
                update_ops.append(mirror_var.assign(updated_value))
        return update_ops

    def _mirror_read_var_ops(self, other):
        """
        Mirror read var ops.

        Args:
            other: Other resource var op.
        """
        assert other in self._mirror_vars
        for old_read_var_op in self._read_var_ops:
            if old_read_var_op == self._this._graph_element.op:
                new_read_var_op = other._graph_element.op
            else:
                new_read_var_op = other.value().op
            self._read_var_ops_mappings[other][old_read_var_op] = new_read_var_op

    def _mirror_all_read_var_ops(self):
        """Mirror all read var ops for each mirror var."""
        for mirror_var in self._mirror_vars:
            self._mirror_read_var_ops(other=mirror_var)

    def _update_consumer(self, other, consumer_op):
        """
        Update consumer.

        Args:
            other: Other resource var op.
            consumer_op: The new consumer.
        """
        old_read_var_op = self._consumer_to_read_var_op[consumer_op]
        new_read_var_op = self._read_var_ops_mappings[other][old_read_var_op]
        update_consumers(
            [consumer_op],
            old_tensor=old_read_var_op.outputs[0],
            new_tensor=new_read_var_op.outputs[0]
        )

    def _update_all_consumers(self):
        """Update all mirror variable consumers."""
        for consumer_op in self._consumer_to_read_var_op:
            if consumer_op in self.mirror_var_init_ops:
                continue
            elif consumer_op.name.startswith(AUTODIST_REPLICA_PREFIX):
                if len(self._mirror_vars) > 1:
                    # Mirror variables are created on GPU,
                    # find one on the same GPU.
                    replica_index = int(consumer_op.name.split(
                        AUTODIST_REPLICA_PREFIX)[1].split('/')[0])

                    self._update_consumer(self._mirror_vars[replica_index], consumer_op)
            else:
                # TODO: Attention: ReadVarOp consumers include the "save".
                print("Consumer %s of value of variable %s is a shared node, do not change to mirror variable"
                      % (consumer_op.name, self._this.op.name))
