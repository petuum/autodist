
from tensorflow.python import ops
from tensorflow.python import constant_op
from autodist.kernel.common import utils


def test_parse_name_scope():
    with ops.Graph().as_default():
        name_scope = 'name_scope/child_name_scope'
        a = constant_op.constant(5)
        new_name = ops.prepend_name_scope(a.name, name_scope)
        assert name_scope == utils.parse_name_scope(new_name)
        print('!!!', a.name, '###', utils.parse_name_scope(a.name))
        assert '' == utils.parse_name_scope(a.name)
