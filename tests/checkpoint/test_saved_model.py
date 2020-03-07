import os
import atexit
from multiprocessing import Process

import numpy as np
import shutil
import tensorflow as tf
from tensorflow.compat.v1 import saved_model
from tensorflow.python.framework import ops

from autodist import AutoDist
from autodist.strategy.all_reduce_strategy import AllReduce
from autodist.checkpoint.saver import Saver as autodist_saver
from autodist.checkpoint.saved_model_builder import SavedModelBuilder


TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000
EPOCHS = 10
inputs = np.random.randn(NUM_EXAMPLES)
noises = np.random.randn(NUM_EXAMPLES)
outputs = inputs * TRUE_W + TRUE_b + noises
resource_spec_file = os.path.join(os.path.dirname(__file__), 'resource_spec.yml')

EXPORT_DIR = "/tmp/autodist_model"
TRAIN_OP_KEY = "saved_model_train_op"
TAG_NAME = "AutoDistCheck"


class MyIterator:
    def initialize(self):
        return tf.zeros(1)

    def get_next(self):
        return inputs


def train_and_save():
    """ Train the model and save the serialized model and its weights. """
    autodist = AutoDist(resource_spec_file, AllReduce(128))
    print('I am going to a scope.')
    with tf.Graph().as_default() as g, autodist.scope():
        x = tf.compat.v1.placeholder(shape=[NUM_EXAMPLES], dtype=tf.float64)
        W = tf.Variable(5.0, name='W', dtype=tf.float64)
        b = tf.Variable(0.0, name='b', dtype=tf.float64)

        def y():
            return W * x + b

        def l(predicted_y, desired_y):
            return tf.reduce_mean(tf.square(predicted_y - desired_y))

        major_version, _, _ = tf.version.VERSION.split('.')
        if major_version == '1':
            optimizer = tf.train.GradientDescentOptimizer(0.01)
        else:
            optimizer = tf.optimizers.SGD(0.01)

        with tf.GradientTape() as tape:
            prediction = y()
            loss = l(prediction, outputs)
            vs = [W, b]
            gradients = tf.gradients(loss, vs)
            train_op = optimizer.apply_gradients(zip(gradients, vs))

        ops.add_to_collection(TRAIN_OP_KEY, train_op)
        fetches = [loss, train_op, b, prediction]
        feeds = [x]

        # NOTE: The AutoDist saver should be declared before the wrapped session.
        saver = autodist_saver()
        session = autodist.create_distributed_session()
        for _ in range(EPOCHS):
            l, _, b, _ = session.run(fetches, feed_dict={feeds[0]: inputs})
            print('node: {}, loss: {}\nb:{}'.format(autodist._cluster.get_local_address(), l, b))
        print('I am out of scope')

        inputs_info = {
            "input_data":
                saved_model.utils.build_tensor_info(feeds[0])
        }
        outputs_info = {
            "loss": saved_model.utils.build_tensor_info(fetches[0]),
            "prediction": saved_model.utils.build_tensor_info(fetches[3])
        }
        serving_signature = saved_model.signature_def_utils.build_signature_def(
            inputs=inputs_info,
            outputs=outputs_info,
            method_name=saved_model.signature_constants.PREDICT_METHOD_NAME
        )
        signature_map = {
            saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                serving_signature,
        }
        if os.path.exists(EXPORT_DIR):
            shutil.rmtree(EXPORT_DIR)
        builder = SavedModelBuilder(EXPORT_DIR)
        builder.add_meta_graph_and_variables(
            sess=session,
            tags=[TAG_NAME],
            saver=saver,
            signature_def_map=signature_map)
        builder.save()


def _get_input_tensor_and_op(graph_input):
    input_op_names = []
    input_tensor_names = []

    for input_item in graph_input.items():
        input_op_name, input_tensor_name = _get_dense_tensor_and_op(input_item)

        input_op_names.append(input_op_name)
        input_tensor_names.append(input_tensor_name)

    return input_op_names, input_tensor_names

def _get_output_tensor_and_op(graph_output):
    output_op_names = []
    output_tensor_names = []

    for output_item in graph_output.items():
        if output_item[1].name != "":
            dense_op_name, dense_tensor_name = _get_dense_tensor_and_op(output_item)
            output_op_names.append(dense_op_name)
            output_tensor_names.append(dense_tensor_name)

    return output_op_names, output_tensor_names

def _get_dense_tensor_and_op(item):
    op_name = item[0]
    tensor_name = item[1].name

    return op_name, tensor_name

def fine_tune():
    print("=====> Validation")
    with tf.compat.v1.Session() as sess:
        assert tf.compat.v1.saved_model.loader.maybe_saved_model_directory(EXPORT_DIR)
        loaded = tf.compat.v1.saved_model.loader.load(sess, [TAG_NAME], EXPORT_DIR)

        train_op = tf.compat.v1.get_collection(TRAIN_OP_KEY)
        serving_signature = loaded.signature_def["serving_default"]

        input_op_names, input_tensor_names = _get_input_tensor_and_op(
            serving_signature.inputs)
        output_op_names, output_tensor_names = _get_output_tensor_and_op(
            serving_signature.outputs)

        input_table = dict(zip(input_op_names, input_tensor_names))
        output_table = dict(zip(output_op_names, output_tensor_names))
        print("Retrieve training operation ...")
        for _ in range(EPOCHS):
            l, _ = sess.run([output_table["loss"], train_op], feed_dict={input_table["input_data"]:inputs})
            print('loss: {}\n'.format(l))
    print("=====> Validation Over")


def save_and_restore():
    def run():
        """This wrapper will handle the AutoDist destructor and garbage collections."""
        try:
            atexit._clear()  # TensorFlow also uses atexit, but running its exitfuncs cause some issues
            train_and_save()
            fine_tune()
        except Exception:
            raise
        finally:
            atexit._run_exitfuncs()


def test_saved_model():
    p = Process(target=save_and_restore)
    p.start()
    p.join()
    if p.exitcode != 0:
        raise SystemExit(f"FAILED running saved_model test")
