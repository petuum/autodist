"""A collection of useful functions used by Replicator."""

from tensorflow.core.framework import variable_pb2
from tensorflow.core.protobuf import meta_graph_pb2, queue_runner_pb2
from tensorflow.python.framework import ops

from autodist.kernel.common.utils import get_op_name, replica_prefix


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
