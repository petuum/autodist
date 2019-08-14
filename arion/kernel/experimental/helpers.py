"""A collection of useful functions used by Replicator."""
from collections import defaultdict

from tensorflow.core.framework import graph_pb2, variable_pb2
from tensorflow.core.protobuf import meta_graph_pb2, queue_runner_pb2
from tensorflow.python.framework import ops
from tensorflow.python.framework.device_spec import DeviceSpecV2

from autodist.const import COLOCATION_PREFIX
from autodist.kernel.common import op_info
from autodist.kernel.common.op_info import UNSTAGE_OP_TYPES, STAGE_OP_TYPES
from autodist.kernel.common.utils import get_op_name, get_consumers, get_ancestors, update_consumers, replica_prefix


class ResourceVariableDistributor:
    """Resource Variable Distributor."""

    def __init__(self, resource_var, mirror_vars):
        self._this = resource_var
        self._read_var_ops = {consumer for consumer in get_consumers(resource_var.op)
                              if consumer.type == "ReadVariableOp"}
        self._consumer_to_read_var_op = {c: o for o in self._read_var_ops for c in get_consumers(o)}
        self._read_var_op_to_consumers = {o: get_consumers(o) for o in self._read_var_ops}
        self._mirror_vars = mirror_vars
        self._read_var_ops_mappings = defaultdict(dict)

    def mirror_read_var_ops(self, other):
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

    def update_consumer(self, other, consumer_op):
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
            old_tensor=old_read_var_op._outputs[0],
            new_tensor=new_read_var_op._outputs[0]
        )


def get_ops_to_replicate(graph_item):
    """
    Get ops to be replicated.

    Args:
        graph_item: the GraphItem object to parse through

    Returns: list
    """
    grad_related = set()
    for grad in graph_item.grad_list:
        if isinstance(grad, ops.IndexedSlices):
            grad_related.add(grad.indices)
            grad_related.add(grad.values)
            grad_related.add(grad.dense_shape)
        elif isinstance(grad, ops.Tensor):
            grad_related.add(grad)
        else:
            raise RuntimeError("Incorrect grad.")

    grads_ancestor_ops = get_ancestors([grad.op for grad in grad_related],
                                       include_control_inputs=True)

    pipeline_ops = graph_item.pipeline_ops(grads_ancestor_ops)

    global_var_related_ops = set()
    for global_var in ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES):
        global_var_related_ops.add(global_var.op)
        global_var_related_ops.add(global_var.initializer)
        # tf1.0
        # global_var_related_ops.add(global_var._snapshot.op)
        # tf2.0, refer to tag:icml autodist/patch.py:
        # global_var_related_ops.add(global_var._graph_element.op)
        # search all read_var_ops
        read_variable_ops = {consumer for consumer in get_consumers(global_var.op) if consumer.type == "ReadVariableOp"}
        global_var_related_ops.update(read_variable_ops)

    table_related_ops = set()
    for table_init in ops.get_collection(ops.GraphKeys.TABLE_INITIALIZERS):
        table_related_ops.add(table_init)
        table_related_ops.add(table_init.inputs[0].op)

    # Assume that all variables are member of either GLOBAL_VARIABLES
    # or LOCAL_VARIABLES.
    local_var_op_to_var = {var.op: var for var in ops.get_collection(ops.GraphKeys.LOCAL_VARIABLES)}
    local_var_ops = set(local_var_op_to_var.keys())
    local_var_ops.intersection_update(grads_ancestor_ops)

    ops_to_replicate = grads_ancestor_ops.copy()
    ops_to_replicate.update(pipeline_ops)

    # var_handles1 = [op for op in ops_to_replicate if op.type == "VarHandleOp"]
    # var_handles2 = [op for op in global_var_related_ops if op.type == "VarHandleOp"]

    ops_to_replicate.difference_update(global_var_related_ops)
    ops_to_replicate.difference_update(table_related_ops)
    ops_to_replicate.update([local_var_op_to_var[var_op].initializer for var_op in local_var_ops])

    return ops_to_replicate


def construct_multi_gpu_graph_def(single_gpu_graph_def, op_names_to_replicate, op_names_to_share, num_replicas,
                                  replica_devices):
    """
    Given a single GPU graph def, construct the graph for multiple GPUs / replicas.

    Args:
        single_gpu_graph_def: The original, single-GPU graph def.
        op_names_to_replicate: The ops to replicate.
        op_names_to_share: The ops to share across replicas.
        num_replicas: The number of replicas.
        replica_devices: The devices to store each replica on.

    Returns:
        GraphDef
    """
    def _update_colocation(node, replica_id=None):
        if '_class' not in node.attr:
            return
        class_list = node.attr['_class'].list
        to_delete = []
        for i in range(len(class_list.s)):
            s = class_list.s[i].decode('utf-8')
            if s.startswith(COLOCATION_PREFIX):
                op_name_to_bind_to = s[len(COLOCATION_PREFIX):]
                if op_name_to_bind_to in op_names_to_replicate:
                    # delete colocation constraint if shared op needs to be
                    # colocated with replica op
                    if replica_id is None:
                        to_delete.append(s)
                    else:
                        new_op_name_to_bind_to = \
                            ops.prepend_name_scope(
                                op_name_to_bind_to,
                                replica_prefix(replica_id))
                        class_list.s[i] = ('%s%s' % (COLOCATION_PREFIX, new_op_name_to_bind_to)).encode('utf-8')
        for item in to_delete:
            class_list.s.remove(item.encode('utf-8'))

    # TODO(Hao): this function is tricky, should infer the cpu device corresponding
    # to this GPU using DeviceSpec
    def _get_cpu_device(device_string):
        default_cpu_device_name = 'CPU:0'
        if 'GPU' in device_string:
            pos = device_string.rfind('/')
            device_name = device_string[:pos] + '/' + default_cpu_device_name
        return device_name

    multi_gpu_graph_def = graph_pb2.GraphDef()
    multi_gpu_graph_def.library.Clear()
    multi_gpu_graph_def.library.CopyFrom(single_gpu_graph_def.library)
    for node in single_gpu_graph_def.node:
        if node.name in op_names_to_share:
            multi_gpu_graph_def.node.extend([node])  # copy
            new_node = multi_gpu_graph_def.node[-1]
            for i in range(len(new_node.input)):
                if get_op_name(new_node.input[i]) in op_names_to_replicate:
                    new_node.input[i] = \
                        ops.prepend_name_scope(new_node.input[i],
                                               replica_prefix(0))
            _update_colocation(new_node)
        elif node.name in op_names_to_replicate:
            for replica_id in range(num_replicas):
                multi_gpu_graph_def.node.extend([node])  # copy
                new_node = multi_gpu_graph_def.node[-1]
                new_node.name = \
                    ops.prepend_name_scope(new_node.name,
                                           replica_prefix(replica_id))
                if 'CPU' not in new_node.device.upper():
                    old_device = DeviceSpecV2.from_string(new_node.device)
                    if new_node.op in op_info.CPU_ONLY_TYPES:
                        new_device = DeviceSpecV2.from_string(_get_cpu_device(replica_devices[replica_id]))
                    else:
                        new_device = DeviceSpecV2.from_string(replica_devices[replica_id])
                    new_node.device = old_device.make_merged_spec(new_device).to_string()
                for i in range(len(new_node.input)):
                    if get_op_name(new_node.input[i]) in op_names_to_replicate:
                        new_node.input[i] = \
                            ops.prepend_name_scope(
                                new_node.input[i],
                                replica_prefix(replica_id))
                if new_node.op in STAGE_OP_TYPES + UNSTAGE_OP_TYPES:
                    new_node.attr['shared_name'].s = (
                        ops.prepend_name_scope(
                            new_node.attr['shared_name'].s,
                            replica_prefix(replica_id))).encode('utf-8')
                _update_colocation(new_node, replica_id)
                if 'frame_name' in new_node.attr:
                    new_node.attr['frame_name'].s = (
                        ops.prepend_name_scope(
                            new_node.attr['frame_name'].s,
                            replica_prefix(replica_id))).encode('utf-8')
        else:
            raise RuntimeError("Should not reach here")

    return multi_gpu_graph_def


def handle_collection_def(multi_gpu_meta_graph_def, op_names_to_replicate, num_replicas):
    """
    Handle collection def.

    Modifies `multi_gpu_meta_graph_def` in place.

    Args:
        multi_gpu_meta_graph_def: Multi GPU MetaGraphDef.
        op_names_to_replicate: The op names to replicate.
        num_replicas: The number of replicas.
    """
    allow_bytes_list_keys = [ops.GraphKeys.QUEUE_RUNNERS,
                             ops.GraphKeys.GLOBAL_VARIABLES,
                             ops.GraphKeys.TRAINABLE_VARIABLES,
                             ops.GraphKeys.MOVING_AVERAGE_VARIABLES,
                             ops.GraphKeys.LOCAL_VARIABLES,
                             ops.GraphKeys.MODEL_VARIABLES,
                             ops.GraphKeys.GLOBAL_STEP]
    keys_to_remove = []
    for key, col_def in multi_gpu_meta_graph_def.collection_def.items():
        kind = col_def.WhichOneof("kind")
        # Update node_list collections (e.g., GLOBAL_STEP, TRAIN_OP, UPDATE_OP,
        # LOSSES, ...)
        if kind == 'node_list':
            new_col_def = _get_new_col_def_of_node_list(
                col_def, op_names_to_replicate, num_replicas)
            multi_gpu_meta_graph_def.collection_def[key].Clear()
            multi_gpu_meta_graph_def.collection_def[key].CopyFrom(new_col_def)
        elif kind == 'bytes_list':
            if ops.get_from_proto_function(key):
                # Collections in allow_bytes_list_keys will be handled
                # explicitly below
                # (e.g., QUEUE_RUNNERS, LOCAL_VARIABLES, ...)
                if key in allow_bytes_list_keys:
                    continue
                # Remove unhandled collections (e.g., COND_CONTEXT)
                # TODO: Handle all protos in ops.GraphKeys
                else:
                    keys_to_remove.append(key)
            # Keep collections without proto function
            # (e.g., user defined string)
            else:
                continue
        else:
            raise RuntimeError("Should not reach here")
    for key in keys_to_remove:
        del multi_gpu_meta_graph_def.collection_def[key]

    # Update QUEUE_RUNNERS and LOCAL_VARIABLES collection
    _update_queue_runners(multi_gpu_meta_graph_def, op_names_to_replicate,
                          num_replicas)
    _update_local_variables(multi_gpu_meta_graph_def, op_names_to_replicate,
                            num_replicas)
    # update_shard_info_for_in_graph(multi_gpu_meta_graph_def, num_replicas)


def _get_new_col_def_of_node_list(col_def, op_names_to_replicate, num_replicas):
    new_col_def = meta_graph_pb2.CollectionDef()
    for tensor_name in col_def.node_list.value:
        if get_op_name(tensor_name) in op_names_to_replicate:
            new_col_def.node_list.value.extend(
                [ops.prepend_name_scope(tensor_name, replica_prefix(i))
                 for i in range(num_replicas)])
        else:
            new_col_def.node_list.value.append(tensor_name)
    return new_col_def


def _update_queue_runners(multi_gpu_meta_graph_def, op_names_to_replicate, num_replicas):
    def _get_new_qr_def(qr_def, prefix, only_rename_enqueue_ops):
        new_qr_def = queue_runner_pb2.QueueRunnerDef()
        new_qr_def.CopyFrom(qr_def)
        del new_qr_def.enqueue_op_name[:]
        for enqueue_op_name in qr_def.enqueue_op_name:
            new_qr_def.enqueue_op_name.append(
                ops.prepend_name_scope(enqueue_op_name, prefix))
        if not only_rename_enqueue_ops:
            new_qr_def.queue_name = \
                ops.prepend_name_scope(qr_def.queue_name, prefix)
            new_qr_def.close_op_name = \
                ops.prepend_name_scope(qr_def.close_op_name, prefix)
            new_qr_def.cancel_op_name = \
                ops.prepend_name_scope(qr_def.cancel_op_name, prefix)
        return new_qr_def

    if ops.GraphKeys.QUEUE_RUNNERS not in multi_gpu_meta_graph_def.collection_def:
        return

    qr_collection = \
        multi_gpu_meta_graph_def.collection_def[ops.GraphKeys.QUEUE_RUNNERS]
    new_qr_col = meta_graph_pb2.CollectionDef()
    for qr_def_string in qr_collection.bytes_list.value:
        qr_def = queue_runner_pb2.QueueRunnerDef()
        qr_def.ParseFromString(qr_def_string)
        assert qr_def.enqueue_op_name
        if qr_def.enqueue_op_name[0] in op_names_to_replicate:
            if qr_def.queue_name in op_names_to_replicate:
                new_qr_defs = \
                    [_get_new_qr_def(qr_def, replica_prefix(i), False)
                     for i in range(num_replicas)]
            else:
                new_qr_defs = \
                    [_get_new_qr_def(qr_def, replica_prefix(i), True)
                     for i in range(num_replicas)]
            new_qr_col.bytes_list.value.extend([new_qr_def.SerializeToString()
                                                for new_qr_def in new_qr_defs])
        else:
            new_qr_col.bytes_list.value.append(qr_def.SerializeToString())
    multi_gpu_meta_graph_def.collection_def[ops.GraphKeys.QUEUE_RUNNERS].Clear()
    multi_gpu_meta_graph_def.collection_def[ops.GraphKeys.QUEUE_RUNNERS] \
        .CopyFrom(new_qr_col)


def _update_local_variables(multi_gpu_meta_graph_def, op_names_to_replicate, num_replicas):
    def _get_new_var_def(var_def, prefix):
        new_var_def = variable_pb2.VariableDef()
        new_var_def.CopyFrom(var_def)
        new_var_def.variable_name = \
            ops.prepend_name_scope(var_def.variable_name, prefix)
        new_var_def.initializer_name = \
            ops.prepend_name_scope(var_def.initializer_name, prefix)
        new_var_def.snapshot_name = \
            ops.prepend_name_scope(var_def.snapshot_name, prefix)
        return new_var_def

    if ops.GraphKeys.LOCAL_VARIABLES not in multi_gpu_meta_graph_def.collection_def:
        return

    lv_collection = \
        multi_gpu_meta_graph_def.collection_def[ops.GraphKeys.LOCAL_VARIABLES]
    new_lv_col = meta_graph_pb2.CollectionDef()
    for var_def_string in lv_collection.bytes_list.value:
        var_def = variable_pb2.VariableDef()
        var_def.ParseFromString(var_def_string)
        if get_op_name(var_def.variable_name) in op_names_to_replicate:
            new_var_defs = \
                [_get_new_var_def(var_def, replica_prefix(i))
                 for i in range(num_replicas)]
            new_lv_col.bytes_list.value.extend(
                [new_var_def.SerializeToString()
                 for new_var_def in new_var_defs])
        else:
            new_lv_col.bytes_list.value.append(var_def.SerializeToString())
    multi_gpu_meta_graph_def.collection_def[ops.GraphKeys.LOCAL_VARIABLES] \
        .Clear()
    multi_gpu_meta_graph_def.collection_def[ops.GraphKeys.LOCAL_VARIABLES] \
        .CopyFrom(new_lv_col)
    if not lv_collection.bytes_list.value:
        del multi_gpu_meta_graph_def \
            .collection_def[ops.GraphKeys.LOCAL_VARIABLES]
