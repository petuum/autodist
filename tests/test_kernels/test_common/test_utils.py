
from tensorflow.python import ops
from tensorflow.python import constant_op
from tensorflow.python.ops.gen_control_flow_ops import no_op

from autodist.kernel.common import utils
from autodist.kernel.common.utils import replica_prefix, strip_replica_prefix


def test_parse_name_scope():
    with ops.Graph().as_default():
        name_scope = 'name_scope/child_name_scope'
        a = constant_op.constant(5)
        new_name = ops.prepend_name_scope(a.name, name_scope)
        assert new_name == 'name_scope/child_name_scope/Const:0'
        assert name_scope == utils.parse_name_scope(new_name)
        assert '' == utils.parse_name_scope(a.name)


        with ops.control_dependencies([no_op(name='my_op')]):
            b = constant_op.constant(6)
        name_scope = 'name_scope'
        new_name = ops.prepend_name_scope(b.op.node_def.input[0], name_scope)
        assert new_name == '^name_scope/my_op'
        assert name_scope == utils.parse_name_scope(new_name)


def test_strip_replica_prefix():
    for name in ['my_op', '^my_op', 'my_tensor:0']:
        new_name = ops.prepend_name_scope(name, replica_prefix(12))
        assert strip_replica_prefix(new_name) == name
