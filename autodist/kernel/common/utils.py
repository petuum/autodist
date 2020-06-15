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

"""A collection of useful functions for the kernel submodule."""
from collections import deque

from tensorflow.core.framework.attr_value_pb2 import AttrValue
from tensorflow.python.util.compat import as_bytes

from autodist.const import AUTODIST_REPLICA_PREFIX, COLOCATION_PREFIX


def get_op_name(tensor_name):
    """
    Given a tensor name, return the corresponding op name.

    Args:
        tensor_name: The name of a tensor.

    Returns:
        str: The op name
    """
    return tensor_name.replace('^', '').split(':')[0]


def strip_replica_prefix(name):
    """
    Given a tensor or op name, strip the AUTODIST-REPLICA prefix if there exists as the prefix.

    Args:
        name (string): op or tensor name

    Returns:
        str: The stripped name
    """
    i = name.find('/')
    if i != -1:
        if AUTODIST_REPLICA_PREFIX in name[:i]:
            return name[i + 1:] if not name.startswith('^') else '^' + name[i + 1:]
    return name


def parse_name_scope(name):
    """
    Given a tensor or op name, return the name scope of the raw name.

    Args:
        name: The name of a tensor or an op.

    Returns:
        str: The name scope
    """
    i = name.rfind('/')
    if i != -1:
        return name[:i] if not name.startswith('^') else name[1:i]
    return ''


def parse_optimizer_scope(update_op_name):
    """
    Given the name of an update_op, return its optimizer name scope.

    Args:
        update_op_name: the name of an update_op (usually ResourceApply).

    Returns:
        str: The outermost name scope of the optimizer
    """
    first_pos = update_op_name.find('/')
    second_pos = update_op_name.find('/', first_pos + 1)
    return update_op_name[:second_pos + 1]


def replica_prefix(replica_id):
    """
    Generate replica prefix based on replica id.

    Examples:
        >>> replica_prefix(1)
        {AUTODIST_REPLICA_PREFIX}1

    Args:
        replica_id (str, int): 0,1,2,3...

    Returns:
        str: The replica prefix
    """
    return f"{AUTODIST_REPLICA_PREFIX}{replica_id}"


def get_consumers(op):
    """
    Get a flat list from [output[0].consumers(), output[1].consumers(), ...].

    Args:
        op: TensorFlow Operator

    Returns:
        List[Operation]: The list of consumers
    """
    return [consumer for output in op.outputs for consumer in output.consumers()]


def get_control_consumers(op):
    """
    Get a flat list of the control-dependency consumers ([B]) of the op (A).

    A: [B]
    A --> B
    B depends on A

    Args:
        op: TensorFlow Operator

    Returns:
        List[Operation]: The list of control-dependency consumers
    """
    return op._control_outputs


def traverse(start_ops, end_ops=None, neighbors_fn=None):
    """
    Traverse a graph and output the visited nodes.

    Args:
        start_ops (iter): The nodes to start the traversal from.
        end_ops (iter): Optional. The nodes at which to stop traversing.
        neighbors_fn (func): Optional. Function from Op -> Iter[Op] that provides the neighbors of an op.
            Defaults to `get_consumers`.

    Returns:
        Set[Operation]: The visited nodes
    """
    end_ops = end_ops or set()
    neighbors_fn = neighbors_fn or get_consumers

    visited = set()
    queue = deque()
    queue.extend(start_ops)

    while queue:
        curr_op = queue.popleft()
        if curr_op in visited:
            continue
        visited.add(curr_op)
        if curr_op in end_ops:
            continue
        queue.extend(neighbors_fn(curr_op))

    return visited


def get_ancestors(start_ops, end_ops=None, include_control_inputs=False):
    """
    Get all ancestor ops of the start ops.

    Starting from start_ops,
    follow the computation graph backwards from consumer to input to find ancestors.
    Stop navigating the graph at end_ops.
    Include both start_ops and end_ops in the returning set of ancestor ops.

    Args:
        start_ops (list, set): The set of ops from which to begin traversing the graph.
        end_ops (list, set): The set of ops at which to stop traversing the graph.
        include_control_inputs: Whether or not to also consider control dependencies as edges.

    Returns:
        Set[Operation]: The ancestor ops
    """
    def get_neighbors(op):
        out = [input_tensor.op for input_tensor in op.inputs]
        if include_control_inputs:
            out.extend(op.control_inputs)
        return out

    return traverse(start_ops, end_ops=end_ops, neighbors_fn=get_neighbors)


def update_consumers(consumers, old_tensor, new_tensor):
    """
    For each consumer's inputs, replace old_tensor with new_tensor.

    Be careful using op.consumers() directly as an argument,
    since this causes incorrect list iteration.

    Args:
        consumers (List[Operation]): The consumer ops in the graph to be modified.
        old_tensor (Tensor): The tensor whose link to its consumer will be removed
        new_tensor (Tensor): The tensor which will replace old_tensor
    """
    for consumer_op in consumers:
        for i, x in enumerate(consumer_op.inputs):
            if x == old_tensor:
                consumer_op._update_input(i, new_tensor)


def update_control_consumers(control_consumer_ops, old_op, new_op):
    """
    For each consumer's control inputs, replace old_op with new_op.

    Args:
        control_consumer_ops (List[Operation]): The control-dep consumer ops in the graph to be modified.
        old_op (Operation): The op whose link to its control-dep consumer will be removed
        new_op (Operation): The op which will replace old_op
    """
    for control_consumer_op in control_consumer_ops:
        control_inputs = list(control_consumer_op.control_inputs)
        size = len(control_inputs)
        control_inputs.remove(old_op)
        assert size - 1 == len(control_inputs)
        control_inputs.append(new_op)
        assert size == len(control_inputs)
        control_consumer_op._remove_all_control_inputs()
        control_consumer_op._add_control_inputs(control_inputs)


def update_colocation_group(ops, old_op, new_op):
    """
    For each op in ops, we replace the colocation group as old_op to colocation group as new_op.

    Args:
        ops (Iterable[Operation]): The operations to update
        old_op (Operation): The op having the old colocation group
        new_op (Operation): The op having the new colocation group
    """
    old_groups = old_op.colocation_groups() or [COLOCATION_PREFIX + as_bytes(new_op.name)]
    new_groups = new_op.colocation_groups() or [COLOCATION_PREFIX + as_bytes(new_op.name)]
    for op in ops:
        if op.colocation_groups() == old_groups:
            op._set_attr("_class", AttrValue(list=AttrValue.ListValue(s=new_groups)))
            assert op.colocation_groups() == new_groups


def remove_from_control_consumers(control_consumer_ops, op_to_remove):
    """
    Remove the op_to_remove from the control inputs for each op in "control_consumer_ops".

    Args:
        control_consumer_ops (List[Operation]): Ops that have op_to_remove as their current control inputs
        op_to_remove (Operation): The op to be removed
    """
    for control_consumer_op in control_consumer_ops:
        control_inputs = list(control_consumer_op.control_inputs)
        size = len(control_inputs)
        control_inputs.remove(op_to_remove)
        assert size - 1 == len(control_inputs)
        control_consumer_op._remove_all_control_inputs()
        control_consumer_op._add_control_inputs(control_inputs)


def get_index_from_tensor_name(tensor_name):
    """
    Get the index of the tensor of a certain op.

    Args:
        tensor_name (str): The tensor name

    Returns:
        int: The index of the tensor
    """
    return int(tensor_name.split(':')[1])
