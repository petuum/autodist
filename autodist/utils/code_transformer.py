"""
Code Transformer.

This allows for transforming user code to avoid
embedding datasets in the TensorFlow graph using
placeholders.
"""
import types
import gast

from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import compiler
from tensorflow.python.autograph.pyct import ast_util


class Transformer(gast.NodeTransformer):
    """Transforms dataset function based on pattern."""

    def __init__(self, pat):
        self._tensor_names = []
        self._pattern = pat
        self._function_depth = 0

    def visit_FunctionDef(self, node):  # noqa: N802
        """Add feed dict to the beginning of the outermost function."""
        def _generate_imports():
            return parser.parse_str("from tensorflow.python.ops import array_ops")

        self._function_depth += 1
        if self._function_depth == 1:
            node.body.insert(0, _generate_imports())
            node.body.insert(0, parser.parse_str("__fd = {}"))
        return self.generic_visit(node)

    def visit_Assign(self, node):  # noqa: N802
        """Add placeholders if we find the provided pattern in an assingment stmt."""
        def _generate_place_stmt(var: str):
            return parser.parse_str(f"{var}_place = array_ops.placeholder({var}.dtype, shape={var}.shape)")

        def _generate_fd_stmt(var: str):
            return parser.parse_str(f"__fd[{var}_place] = {var}")

        def _generate_cond_call(vars_):
            cond = "all(("
            for var in vars_:
                cond += f"isinstance({var}, np.ndarray),"
            cond += "))"
            return parser.parse_str(cond).value  # gast.Call

        if (self._function_depth == 1) and ast_util.matches(node.value, self._pattern):
            # extract ndarrays to be replaced
            self._tensor_names = [x.id for x in node.value.args[0].elts]
            body = []
            new_node = ast_util.copy_clean(node)
            for i, t in enumerate(self._tensor_names):
                new_node.value.args[0].elts[i].id = f"{t}_place"
                body.append(_generate_place_stmt(t))
                body.append(_generate_fd_stmt(t))
            body.append(new_node)
            # put placeholder code inside of a If block
            return gast.If(_generate_cond_call(self._tensor_names), body, [node])
        else:
            return node

    def visit_Return(self, node):  # noqa: N802
        """Return the feed dict from the outermost function."""
        if self._function_depth == 1:
            elts = [node.value, gast.Name(id="__fd", ctx=gast.Load(), annotation=None)]
            node.value = gast.Tuple(elts=elts, ctx=gast.Load())
        self._function_depth -= 1
        return node


def transform(dataset_fn):
    """Transforms function dataset_fn."""
    assert isinstance(dataset_fn, types.FunctionType)

    # parse, get ast
    # pylint: disable=unexpected-keyword-arg
    node, _ = parser.parse_entity(dataset_fn, future_features=())

    # transform ast if we find required pattern
    pattern = parser.parse_expression("tf.data.Dataset.from_tensor_slices(_)")
    tr = Transformer(pattern)
    xform_node = tr.visit(node)
    gast.fix_missing_locations(xform_node)

    # convert it back to a callable
    # pylint: disable=unbalanced-tuple-unpacking
    module, _, _ = compiler.ast_to_object(xform_node, include_source_map=True)

    globs = dataset_fn.__globals__
    name = dataset_fn.__name__
    code = getattr(module, name).__code__

    bound = types.FunctionType(code=code, globals=globs, name=name, argdefs=(), closure=())
    return bound
