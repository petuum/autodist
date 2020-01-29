"""Common helpers for resource variables."""

from tensorflow.python.eager import context, tape
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops.resource_variable_ops import _maybe_set_handle_data

from autodist.kernel.common.utils import get_consumers


def get_read_var_ops(var_handle_op, exclude_snapshot=False):
    """
    Given a resource handle op, get all its read variable ops.

    Args:
        var_handle_op: the varhandleop of the resource variable of interest.
        exclude_snapshot (False): whether to skip the default ReadVariableOp bundled (i.e. "/Read/ReadVariableOp").

    Returns:
        list: list of ReadVariableOps of it.
    """
    read_var_ops = {consumer for consumer in get_consumers(var_handle_op) if consumer.type == "ReadVariableOp"}
    if exclude_snapshot:
        read_var_ops = {op for op in read_var_ops if not op.name.endswith("/Read/ReadVariableOp")}
    return read_var_ops


def gen_read_var_op(var_handle_op, dtype):
    """
    Given a resource handle and dtype, generate a read variable op.

    Copied from ResourceVariable.read_variable_op().
    """
    result = gen_resource_variable_ops.read_variable_op(var_handle_op, dtype)
    _maybe_set_handle_data(dtype, var_handle_op, result)
    if not context.executing_eagerly():
        # Note that if a control flow context is active the input of the read op
        # might not actually be the handle. This line bypasses it.
        tape.record_operation(
            "ReadVariableOp", [result], [var_handle_op], lambda x: [x])
    return result


def get_read_var_tensor(var_handle_op):
    """Given a resource handle, get the tensor of its default readable value."""
    if var_handle_op.type == 'VarHandleOp':
        for read_var_op in get_read_var_ops(var_handle_op):
            if read_var_op.name.endswith("/Read/ReadVariableOp"):
                return read_var_op.outputs[0]
    if len(var_handle_op.outputs) > 1:
        raise ValueError("Can't get the variable reading tensor from '{}'. "
                         "It may not be a proper variable op.".format(var_handle_op.name))
    return var_handle_op.outputs[0]
