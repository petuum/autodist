"""Implementations for the proxy variable in PS."""
from collections import defaultdict

from tensorflow.python import ops
from tensorflow.python.ops import variable_scope

from tensorflow.python.util.compat import as_bytes
from tensorflow.core.framework.attr_value_pb2 import AttrValue as pb2_AttrValue

from autodist.const import COLOCATION_PREFIX
from autodist.kernel.common.resource_variable_utils import get_read_var_ops, get_read_var_tensor, gen_read_var_op
from autodist.kernel.common.utils import get_consumers, update_consumers, replica_prefix, AUTODIST_REPLICA_PREFIX
from autodist.utils import logging


# Hao: might be useful in the future..
# def gen_mirror_var_init_op(variable_replicators):
#     """Given a list of variable replicators, generate an "initialize" op to trigger the AssignOps of all."""
#     all_mirror_var_init_ops = sum([m.mirror_var_init_ops for m in variable_replicators if m], [])
#     if all_mirror_var_init_ops:
#         with ops.control_dependencies(all_mirror_var_init_ops):
#             gen_control_flow_ops.no_op(name=InitOps.MIRROR_VARIABLE_INIT_OP.value)


# pylint: disable=too-many-instance-attributes
class ProxyVariable:
    """Proxy variable implementation."""

    def __init__(self, resource_var, graph_item, proxy_device):
        self._graph_item = graph_item
        self._initial_value = graph_item.graph.get_tensor_by_name(resource_var.initial_value.name)
        self._dtype = resource_var.dtype.base_dtype
        self._this_op = graph_item.graph.get_operation_by_name(resource_var.op.name)
        self._read_var_ops = get_read_var_ops(self._this_op)
        self._consumer_to_read_var_op = {c: o for o in self._read_var_ops for c in get_consumers(o)}
        self._read_var_op_to_consumers = {o: get_consumers(o) for o in self._read_var_ops}

        self._proxy_vars = []
        self._proxy_var_init_ops = []
        self._read_var_ops_mappings = defaultdict(dict)

        self._build_proxy_on(proxy_device)

    def _build_proxy_on(self, destination_device):
        """
        Build a proxy of the original variable on `destination_device`.

        Args:
            destination_device: the destination device where the proxy is on.
        """
        is_gpu = destination_device.device_type.upper() == 'GPU' if destination_device.device_type else False
        prefix = replica_prefix(destination_device.device_index) if is_gpu else replica_prefix('CPU')
        with ops.device(destination_device):
            mirror_var = variable_scope.get_variable(
                ops.prepend_name_scope(self._this_op.name, prefix),
                dtype=self._dtype,
                initializer=self._initial_value,
                trainable=False
            )
        self._graph_item.info.update(variables=[mirror_var], replace=False)  # Should we update graph_item.info?
        self._proxy_vars.append(mirror_var)
        self._proxy_var_init_ops.append(mirror_var.assign(get_read_var_tensor(self._this_op)))
        self._mirror_all_read_var_ops()
        self._update_all_consumers()

    def get_all_update_ops(self, grad_apply_finished, worker_device=None):
        """Create and return new update ops for mirror vars."""
        with ops.device(worker_device):
            with ops.control_dependencies(grad_apply_finished):
                updated_value = gen_read_var_op(self._this_op.outputs[0], self._dtype)  # create new read var op
        update_ops = []
        for proxy_var in self._proxy_vars:
            with ops.device(proxy_var.device):
                update_ops.append(proxy_var.assign(updated_value))
        return update_ops

    def update_colocation_group(self, get_colocation_op):
        """Update operations colocated with master variables to be colocated with mirror variables."""
        for op in self._graph_item.graph.get_operations():
            # Do not update shared node
            # Do not update operations within the variable scope of master var
            # Do not update the VarhandleOp itself
            if not op.name.startswith(AUTODIST_REPLICA_PREFIX) or \
                    op.name.startswith(self._this_op.name + '/') or \
                    (op.name.startswith(self._this_op.name) and op.type == 'VarHandleOp'):
                continue
            new_colocation_group = []
            for colocation_group in op.colocation_groups():
                current_binding_op = get_colocation_op(colocation_group)
                if current_binding_op == self._this_op:
                    op_name_to_bind_to = (COLOCATION_PREFIX + as_bytes(self._proxy_vars[0].op.name))
                    new_colocation_group.append(op_name_to_bind_to)
                else:
                    new_colocation_group.append(colocation_group)
            op._set_attr("_class", pb2_AttrValue(list=pb2_AttrValue.ListValue(s=new_colocation_group)))

    def _mirror_read_var_ops(self, other):
        """
        Mirror read var ops.

        Args:
            other: Other resource var op.
        """
        assert other in self._proxy_vars
        for old_read_var_op in self._read_var_ops:
            if old_read_var_op == get_read_var_tensor(self._this_op).op:
                new_read_var_op = other._graph_element.op
            else:
                new_read_var_op = other.value().op
            self._read_var_ops_mappings[other][old_read_var_op] = new_read_var_op

    def _mirror_all_read_var_ops(self):
        """Mirror all read var ops for each mirror var."""
        for mirror_var in self._proxy_vars:
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
            if consumer_op in self._proxy_var_init_ops:
                continue
            elif consumer_op.name.startswith(AUTODIST_REPLICA_PREFIX):
                if len(self._proxy_vars) > 1:
                    raise ValueError('Now we only create one proxy per variable at most...')
                self._update_consumer(self._proxy_vars[0], consumer_op)
            else:
                # TODO: Attention: ReadVarOp consumers include the "save".
                logging.warning("Consumer %s of value of variable %s is a shared node, do not change to mirror variable"
                                % (consumer_op.name, self._this_op.name))
