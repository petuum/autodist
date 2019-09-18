import tensorflow as tf
import numpy as np

from autodist.utils.code_transformer import transform
from tensorflow.python.framework import ops

def prepare_mnist_features_and_labels(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)
    return x, y

def dataset_fn_0():
    x, y = np.random.rand(10, 28, 28, 1), np.random.rand(10,)

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(prepare_mnist_features_and_labels)
    ds = ds.take(20000).shuffle(20000).batch(32)
    return ds

def dataset_fn_1():
    x, y = tf.ones(shape=(10, 28, 28, 1)), tf.ones(shape=(10,))  # Tensors are not converted

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(prepare_mnist_features_and_labels)
    ds = ds.take(20000).shuffle(20000).batch(32)
    return ds

def dataset_fn_2():
    x, y = np.random.rand(10, 28, 28, 1), np.random.rand(10,)

    if True:  # nesting allowed
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        ds = ds.map(prepare_mnist_features_and_labels)
        ds = ds.take(20000).shuffle(20000).batch(32)
        return ds

    return None

def dataset_fn_3():
    x, y = np.random.rand(10, 28, 28, 1), np.random.rand(10,)

    def foo():
        ds2 = tf.data.Dataset.from_tensor_slices((x, y))  # we skip this
        return 1, 2
    _, _ = foo()

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(prepare_mnist_features_and_labels)
    ds = ds.take(20000).shuffle(20000).batch(32)
    return ds

def call(fn):  # (() -> ds) -> ds, fd
    with ops.Graph().as_default():
        return transform(fn)()

def assert_fd_ok(fd):
    assert len(fd) == 2
    for k, v in fd.items():
        assert "Placeholder" in k.name

def test_code_transformer():
    _, fd = call(dataset_fn_0)
    assert_fd_ok(fd)

    _, fd = call(dataset_fn_1)
    assert fd == {}

    _, fd = call(dataset_fn_2)
    assert_fd_ok(fd)

    _, fd = call(dataset_fn_3)
    assert_fd_ok(fd)

if __name__ == "__main__":
    test_code_transformer()

