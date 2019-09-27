"""A collection of useful functions for the kernel submodule."""
from collections import defaultdict, deque

from autodist.const import AUTODIST_REPLICA_PREFIX
from autodist.kernel.common.op_info import STAGE_OP_TYPES


def get_op_name(tensor_name):
    """
    Given a tensor name, return the corresponding op name.

    Args:
        tensor_name: The name of a tensor.

    Returns:
        str
    """
    return tensor_name.replace('^', '').split(':')[0]


def parse_name_scope(name):
    """
    Given a tensor or op name, return the name scope of the raw name.

    Args:
        name: The name of a tensor or an op.

    Returns:
        str
    """
    i = name.rfind('/')
    if i != -1:
        return name[:i]
    return ''


def replica_prefix(replica_id):
    """
    Generate replica prefix based on replica id.

    Examples:
        1 -> '<AUTODIST_REPLICA_PREFIX>1'

    Args:
        replica_id (str, int): 0,1,2,3...

    Returns: str
    """
    return f"{AUTODIST_REPLICA_PREFIX}{replica_id}"


def get_consumers(op):
    """
    Get a flat list from [output[0].consumers(), output[1].consumers(), ...].

    Args:
        op: TensorFlow Operator

    Returns: List
    """
    return [consumer for output in op.outputs for consumer in output.consumers()]


# def get_control_consumers(op):
#     """
#     Get a flat list of the control-dependency consumers ([B]) of the op (A):
#
#     A: [B]
#     A --> B
#     B depends on A
#
#     Args:
#         op: TensorFlow Operator
#
#     Returns:
#         List
#     """
#     return op._control_outputs


def traverse(start_ops, end_ops=None, neighbors_fn=None):
    """
    Traverse a graph and output the visited nodes.

    Args:
        start_ops (iter): The nodes to start the traversal from.
        end_ops (iter): Optional. The nodes at which to stop traversing.
        neighbors_fn (func): Optional. Function from Op -> Iter[Op] that provides the neighbors of an op.
            Defaults to `get_consumers`.

    Returns:
        Set[Op]
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
        Set
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
        consumers (list): must be the reference to the consumer ops in the graph to be modified.
        old_tensor (Tensor): the tensor whose link to its consumer will be removed
        new_tensor (Tensor): the tensor which will replace old_tensor
    """
    for consumer_op in consumers:
        for i, x in enumerate(consumer_op.inputs):
            if x == old_tensor:
                consumer_op._update_input(i, new_tensor)


def update_control_consumers(control_consumer_ops, old_op, new_op):
    """
    For each consumer's control inputs, replace old_op with new_op.

    Args:
        control_consumer_ops (list): must be the reference to the control-dep consumer ops in the graph to be modified.
        old_op (Operation): the op whose link to its control-dep consumer will be removed
        new_op (Operation): the op which will replace old_op
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


def get_shared_name_to_stage_ops(input_ops):
    """Get shared_name_to_stage_ops mapping."""
    stage_ops = [op for op in input_ops if op.type in STAGE_OP_TYPES]
    shared_name_to_stage_ops = defaultdict(list)
    for stage_op in stage_ops:
        shared_name = stage_op.get_attr("shared_name")
        shared_name_to_stage_ops[shared_name].append(stage_op)
    return shared_name_to_stage_ops


def get_index_from_tensor_name(tensor_name):
    """Get the index of the tensor of a certain op."""
    return int(tensor_name.split(':')[1])
