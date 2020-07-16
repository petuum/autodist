import os
import atexit
import numpy as np
import tensorflow as tf
from multiprocessing import Process

from autodist import AutoDist
from autodist.checkpoint.saver import Saver as autodist_saver


filepath = os.path.join(os.path.dirname(__file__), 'resource_spec.yml')
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images[:512, :, :, None]
test_images = test_images[:512, :, :, None]
train_labels = train_labels[:512]
test_labels = test_labels[:512]
train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)
checkpoint_dir = '/tmp/ckpt/'

BATCH_SIZE = 64
EPOCHS = 1


def clean_up(fn):
    def run():
        try:
            atexit._clear()
            fn()
        except Exception:
            raise
        finally:
            atexit._run_exitfuncs()
    return run


def train_step(inputs):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD()

    x, y = inputs
    y_hat = model(x, training=True)
    loss = loss_fn(y, y_hat)
    all_vars = []
    for v in model.trainable_variables:
        all_vars.append(v)
    grads = tf.gradients(loss, all_vars)
    update = optimizer.apply_gradients(zip(grads, all_vars))

    return loss, update


@clean_up
def train_and_save():
    autodist = AutoDist(resource_spec_file=filepath)
    with tf.Graph().as_default(), autodist.scope():
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (train_images, train_labels)).repeat(EPOCHS).shuffle(len(train_images)//2).batch(BATCH_SIZE)

        train_iterator = tf.compat.v1.data.make_one_shot_iterator(train_dataset).get_next()
        fetches = train_step(train_iterator)

        saver = autodist_saver()
        sess = autodist.create_distributed_session()
        for step in range(min(10, len(train_images) // BATCH_SIZE * EPOCHS)):
            loss, _ = sess.run(fetches)
            print(f"train_loss: {loss}")
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_suffix = 'mnist'
        checkpoint_name = checkpoint_dir + checkpoint_suffix
        saver.save(sess, checkpoint_name, global_step=step)
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
        print('Checkpoint saved at {%s}' % checkpoint_name)


@clean_up
def fine_tune():
    with tf.compat.v1.Session() as sess:
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (train_images, train_labels)).repeat(EPOCHS).shuffle(len(train_images)//2).batch(BATCH_SIZE)

        train_iterator = tf.compat.v1.data.make_one_shot_iterator(train_dataset).get_next()
        fetches = train_step(train_iterator)

        tf_saver = tf.compat.v1.train.Saver()
        tf_saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
        for _ in range(min(10, len(train_images) // BATCH_SIZE * EPOCHS)):
            loss, _ = sess.run(fetches)
            print(f"train_loss: {loss}")


def test_keras_saver():
    p = Process(target=train_and_save)
    p.start()
    p.join()
    if p.exitcode != 0:
        raise SystemExit(f"FAILED running train and save test")

    p = Process(target=fine_tune)
    p.start()
    p.join()
    if p.exitcode != 0:
        raise SystemExit(f"FAILED running restore test")