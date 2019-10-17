"""Common helpers for resource variables."""

from collections import defaultdict

from tensorflow.core.framework.attr_value_pb2 import AttrValue as pb2_AttrValue
from tensorflow.python import ops
from tensorflow.python.eager import context, tape
from tensorflow.python.framework import device_spec
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.resource_variable_ops import _maybe_set_handle_data
from tensorflow.python.util.compat import as_bytes

from autodist.const import InitOps, COLOCATION_PREFIX
from autodist.kernel.common.utils import get_consumers, update_consumers, replica_prefix, AUTODIST_REPLICA_PREFIX
from autodist.utils import logging


def get_read_var_ops(var_handle_op):
    """Given a resource handle op, get all its read variable ops."""
    return {consumer for consumer in get_consumers(var_handle_op) if consumer.type == "ReadVariableOp"}


def gen_read_var_op(var_handle_op, dtype):
    """
    Given a resource handle and dtype, generate a read variable op.

    Copied from ResourceVariable.read_variable_op().
    """
    result = gen_resource_variable_ops.read_variable_op(var_handle_op,
                                                        dtype)
    _maybe_set_handle_data(dtype, var_handle_op, result)
    if not context.executing_eagerly():
        # Note that if a control flow context is active the input of the read op
        # might not actually be the handle. This line bypasses it.
        tape.record_operation(
            "ReadVariableOp", [result], [var_handle_op], lambda x: [x])
    return result


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


# pylint: disable=too-many-instance-attributes
class ResourceVariableReplicator:
    """Resource Variable Replicator."""

    def __init__(self, resource_var, graph_item):
        self._graph_item = graph_item
        self._initial_value = graph_item.graph.get_tensor_by_name(resource_var.initial_value.name)
        self._dtype = resource_var.dtype.base_dtype
        self._this_op = graph_item.graph.get_operation_by_name(resource_var.op.name)

        self._read_var_ops = get_read_var_ops(self._this_op)
        self._consumer_to_read_var_op = {c: o for o in self._read_var_ops for c in get_consumers(o)}
        self._read_var_op_to_consumers = {o: get_consumers(o) for o in self._read_var_ops}
        self._mirror_vars = []
        self.mirror_var_init_ops = []
        self._read_var_ops_mappings = defaultdict(dict)

    def build_mirror_vars(self, mirror_var_device, num_replicas):
        """Build mirror vars for the master var."""
        master_var = self._this_op
        original_var_device = device_spec.DeviceSpecV2.from_string(master_var.device)
        mv_mapping = self._mirror_vars
        init_ops = self.mirror_var_init_ops

        is_gpu = original_var_device.device_type.upper() == 'GPU' if original_var_device.device_type else False
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
                    ops.prepend_name_scope(self._this_op.name, prefix),
                    dtype=self._dtype,
                    initializer=self._initial_value,
                    trainable=False
                )
                self._graph_item.info.update(variables=[mirror_var], replace=False)
            mv_mapping.append(mirror_var)
            init_ops.append(mirror_var.assign(get_read_var_tensor(self._this_op)))

        self._mirror_all_read_var_ops()
        self._update_all_consumers()
        return self

    def get_all_update_ops(self, grad_apply_finished, worker_device=None):
        """Create and return new update ops for mirror vars."""
        with ops.device(worker_device):
            with ops.control_dependencies(grad_apply_finished):
                updated_value = gen_read_var_op(self._this_op.outputs[0], self._dtype)  # create new read var op
        update_ops = []
        for mirror_var in self._mirror_vars:
            with ops.device(mirror_var.device):
                update_ops.append(mirror_var.assign(updated_value))
        return update_ops

    def update_colocation_group(self, get_colocation_op):
        """Update operations colocated with master variables to be colocated with mirror variables."""
        # Do not update shared node
        if not self._this_op.name.startswith(AUTODIST_REPLICA_PREFIX):
            return
        new_colocation_group = []
        for colocation_group in self._this_op.colocation_groups():
            current_binding_op = get_colocation_op(colocation_group)
            if current_binding_op in self._mirror_vars:
                replica_index = 0
                if len(self._mirror_vars) > 1:
                    # Mirror variables are created on GPU, find one on the same GPU
                    replica_index = int(self._this_op.name.split(AUTODIST_REPLICA_PREFIX)[1].split('/')[0])
                op_name_to_bind_to = (COLOCATION_PREFIX + as_bytes(self._mirror_vars[replica_index].op.name))
                new_colocation_group.append(op_name_to_bind_to)
            else:
                new_colocation_group.append(colocation_group)
        self._this_op._set_attr("_class", pb2_AttrValue(list=pb2_AttrValue.ListValue(s=new_colocation_group)))

    def _mirror_read_var_ops(self, other):
        """
        Mirror read var ops.

        Args:
            other: Other resource var op.
        """
        assert other in self._mirror_vars
        for old_read_var_op in self._read_var_ops:
            if old_read_var_op == get_read_var_tensor(self._this_op).op:
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
                logging.warning("Consumer %s of value of variable %s is a shared node, do not change to mirror variable"
                                % (consumer_op.name, self._this_op.name))
