import numpy as np
import pytest
import tensorflow as tf
from tensorflow.python import ops
from tensorflow.python.keras.optimizer_v2 import adadelta
from tensorflow.python.keras.optimizer_v2 import adagrad
from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.keras.optimizer_v2 import adamax
from tensorflow.python.keras.optimizer_v2 import ftrl
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.keras.optimizer_v2 import nadam
from tensorflow.python.keras.optimizer_v2 import rmsprop
from tensorflow.python.training.training_util import get_or_create_global_step

from autodist import graph_item
from autodist.kernel.common.op_info import UPDATE_OP_VAR_POS


def model_simple():
    with ops.Graph().as_default() as g:
        _TRUE_W = 3.0
        _TRUE_b = 2.0
        _NUM_EXAMPLES = 1000
        inputs = np.random.randn(_NUM_EXAMPLES)
        noises = np.random.randn(_NUM_EXAMPLES)
        desired_y = inputs * _TRUE_W + _TRUE_b + noises

        W = tf.Variable(5.0, name='W', dtype=tf.float64)
        b = tf.Variable(0.0, name='b', dtype=tf.float64)
        variables = [W, b]
        with tf.GradientTape() as tape:
            predicted_y = W * inputs + b
            loss = tf.reduce_mean(tf.square(predicted_y - desired_y))
            gradients = tape.gradient(loss, variables)
    return g, gradients, variables


def model_keras_dense_and_sparse():
    _NUM_LEARNERS = 3
    inputs = []
    intermediates = []
    for _ in range(_NUM_LEARNERS):
        inp = tf.keras.layers.Input(shape=(1,), dtype=tf.dtypes.int32)
        layer = tf.keras.layers.Embedding(1, 4)(inp)
        layer = tf.keras.layers.Dense(1)(layer)
        inputs.append(inp)
        intermediates.append(layer)
    layer = tf.keras.layers.Concatenate(axis=-1)(intermediates)
    layer = tf.keras.layers.Dense(1)(layer)
    return tf.keras.models.Model(inputs, layer)


@pytest.mark.parametrize(
    argnames='optimizer_class, kwargs',
    argvalues=[
        (adadelta.Adadelta, None),
        (adagrad.Adagrad, None),
        (adam.Adam, None),
        (adam.Adam, dict(amsgrad=True)),
        (adamax.Adamax, None),
        (ftrl.Ftrl, None),
        (ftrl.Ftrl,
         dict(l2_shrinkage_regularization_strength=0.1)),
        (gradient_descent.SGD, None),
        (gradient_descent.SGD, dict(momentum=0.5)),
        (nadam.Nadam, None),
        (rmsprop.RMSprop, None),
        (rmsprop.RMSprop, dict(centered=True)),
        (rmsprop.RMSprop, dict(momentum=0.5)),
        (rmsprop.RMSprop,
         dict(momentum=0.5, centered=True)),
    ]
)
def test_update_ops_for_optimizers(optimizer_class, kwargs):
    item = graph_item.GraphItem(graph=ops.Graph())
    with item.as_default():
        model = model_keras_dense_and_sparse()
        trainable_variables = model.trainable_variables
        kwargs = kwargs or {}
        optimizer = optimizer_class(**kwargs)
        print(optimizer)
        grads = optimizer.get_gradients(model.outputs[0], trainable_variables)
        optimizer.apply_gradients(zip(grads, trainable_variables))
        assert len(item.var_op_name_to_grad_info) == len(trainable_variables)


def test_graph_item_context_scope():
    g1 = ops.Graph()
    i1 = graph_item.GraphItem(graph=g1)
    assert graph_item._default_graph_item is None
    with i1.as_default() as item:
        assert graph_item._default_graph_item == i1
        assert item._graph == g1
        assert ops.get_default_graph() == g1
        setattr(item, 'new_attr', 'new_value')
    assert graph_item._default_graph_item is None
    assert getattr(i1, 'new_attr') == 'new_value'


def test_copy():
    g1 = graph_item.GraphItem(graph=ops.Graph())
    with g1.as_default():
        model = model_keras_dense_and_sparse()
        trainable_variables = model.trainable_variables
        optimizer = adagrad.Adagrad()
        grads = optimizer.get_gradients(model.outputs[0], trainable_variables)
        optimizer.apply_gradients(zip(grads, trainable_variables))
    g2 = g1.copy()

    def compare(g1, g2):
        gd1 = g1.graph.as_graph_def()
        gd2 = g2.graph.as_graph_def()
        assert gd1 is not gd2
        d1 = {n.name: n for n in gd1.node}
        d2 = {n.name: n for n in gd2.node}
        assert d1 == d2
        assert g1._grad_target_pairs == g2._grad_target_pairs
        assert g1.info.variables == g2.info.variables
    compare(g1, g2)
