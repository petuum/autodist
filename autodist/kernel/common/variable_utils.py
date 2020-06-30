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

"""Common helpers for `ResourceVariables` and `RefVariables`."""

from tensorflow.python.eager import context, tape
from tensorflow.python.ops import gen_resource_variable_ops, array_ops
from tensorflow.python.ops.resource_variable_ops import _maybe_set_handle_data

from autodist.kernel.common.utils import get_consumers


def is_read_var_op(op, version=None):
    """
    Determines if an op is a read var op.

    Checks if it is a `ReadVariableOp` or an `IdentityOp`. This is because
    `ResourceVariables` use `ReadVariableOps` and `RefVariables` use
    `IdentityOps`.

    Args:
        op (Operation): the operation to inspect
        version (int): TF major version integer

    Returns:
        bool: Whether or not the op is a read var op
    """
    if version == 1:
        return op.type == 'Identity'
    elif version == 2:
        return op.type == 'ReadVariableOp'
    elif version is None:
        return op.type == "ReadVariableOp" or op.type == 'Identity'
    raise ValueError('verion=1 or version=2 or version is None')


def get_read_var_ops(var_op, exclude_snapshot=False):
    """
    Given a resource handle op, get all its read variable ops.

    Args:
        var_op (Operation): VarHandleOp for ResourceVariable, VariableV2 or Variable for RefVariable
        exclude_snapshot (bool): whether to skip the default ReadVariableOp bundled (i.e. "/Read/ReadVariableOp").
            no extra snapshot to exclude for RefVariable, even if `exclude_snapshot=True`.

    Returns:
        List[Operation]: List of read var ops of it.
    """
    read_var_ops = {
        consumer for consumer in get_consumers(var_op)
        if is_read_var_op(consumer)
    }
    if exclude_snapshot:
        read_var_ops = {op for op in read_var_ops if not op.name.endswith("/Read/ReadVariableOp")}
    return read_var_ops


def get_read_var_tensor(var_op):
    """
    Given a var op, get the tensor of its default readable value.

    Args:
        var_op (Operation): the variable op

    Returns:
        Tensor: The tensor of its default readable value
    """
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
    Given a var op, generate the op for reading its value.

    Args:
        var_op (Operation): The var op
        dtype (dtype): The dtype of the data to read

    Returns:
        Operation: The value-reading operation
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
