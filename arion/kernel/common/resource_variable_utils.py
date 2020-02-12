"""Common helpers for resource variables."""

from tensorflow.python.eager import context, tape
from tensorflow.python.ops import gen_resource_variable_ops, array_ops
from tensorflow.python.ops.resource_variable_ops import _maybe_set_handle_data

from autodist.kernel.common.utils import get_consumers


def is_read_var_op(op):
    """Is ReadVariableOp for ResourceVariable, Identity for RefVariable."""
    return op.type == "ReadVariableOp" or op.type == 'Identity'


def get_read_var_ops(var_op, exclude_snapshot=False):
    """
    Given a resource handle op, get all its read variable ops.

    Args:
        var_op: VarHandleOp for ResourceVariable, VariableV2 or Variable for RefVariable
        exclude_snapshot (False): whether to skip the default ReadVariableOp bundled (i.e. "/Read/ReadVariableOp").
            no extra snapshot to exclude for RefVariable, even if `exclude_snapshot=True`.

    Returns:
        list: list of read var ops of it.
    """
    read_var_ops = {
        consumer for consumer in get_consumers(var_op)
        if is_read_var_op(consumer)
    }
    if exclude_snapshot:
        read_var_ops = {op for op in read_var_ops if not op.name.endswith("/Read/ReadVariableOp")}
    return read_var_ops


def get_read_var_tensor(var_op):
    """Given a var op, get the tensor of its default readable value."""
    if var_op.type == 'VarHandleOp':
        for read_var_op in get_read_var_ops(var_op):
            if read_var_op.name.endswith("/Read/ReadVariableOp"):
                return read_var_op.outputs[0]
    elif var_op.type == 'VariableV2' or is_read_var_op(var_op):
        return var_op.outputs[0]
    raise ValueError("Can't get the variable reading tensor from '{}'. "
                     "It may not be a proper variable op.".format(var_op.name))


def gen_read_var_op(var_op, dtype):
    """
    Given a var op, generate the op of value reading.

    ResourceVariable.read_variable_op() or RefVariable.read_value()
    """
    var_op_tensor = var_op.outputs[0]
    if var_op.type == 'VarHandleOp':
        result = gen_resource_variable_ops.read_variable_op(var_op_tensor, dtype)
        _maybe_set_handle_data(dtype, var_op_tensor, result)
        if not context.executing_eagerly():
            # Note that if a control flow context is active the input of the read op
            # might not actually be the handle. This line bypasses it.
            tape.record_operation(
                "ReadVariableOp", [result], [var_op_tensor], lambda x: [x])
        return result
    elif var_op.type == 'VariableV2' or is_read_var_op(var_op):
        return array_ops.identity(var_op_tensor)
    raise ValueError("Can't generate the variable reading tensor from '{}'. "
                     "It may not be a proper variable op.".format(var_op.name))
