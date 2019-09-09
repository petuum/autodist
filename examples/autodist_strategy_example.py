import sys

import numpy as np
import os
import tensorflow as tf

from autodist import AutoDist
from tensorflow.python.training.training_util import get_or_create_global_step

resource_spec_file = os.path.join(os.path.dirname(__file__), 'resource_spec.yml')


def main(_):
    autodist = AutoDist(resource_spec_file, 'PS')

    d = autodist

    fashion_mnist = tf.keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images[:, :, :, None]
    test_images = test_images[:, :, :, None]
    train_labels = train_labels[:]
    test_labels = test_labels[:]
    print(train_images.shape, train_labels.shape)

    train_images = train_images / np.float32(255)
    test_images = test_images / np.float32(255)

    BUFFER_SIZE = len(train_images)

    BATCH_SIZE = 32

    EPOCHS = 2

    with d.scope():


        train_dataset = tf.data.Dataset.from_tensor_slices(
            (train_images, train_labels)).repeat(EPOCHS).shuffle(
            BUFFER_SIZE).batch(BATCH_SIZE)

        train_iterator = tf.compat.v1.data.make_one_shot_iterator(train_dataset).get_next()


        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(2048, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        # optimizer.iterations = get_or_create_global_step()

        def train_step(inputs):

            x, y = inputs
            with tf.GradientTape() as tape:
                y_hat = model(x, training=True)
                loss = loss_fn(y, y_hat)
                all_vars = []
                for v in model.trainable_variables:
                    all_vars.append(v)
                # grads = tape.gradient(loss, all_vars)
                grads = tf.gradients(loss, all_vars)
            update = optimizer.apply_gradients(zip(grads, all_vars))

            return loss, update, optimizer.iterations

        while True:
            loss, _, i = d.run(train_step, train_iterator)
            print(f"step: {i}, train_loss: {loss}")


main(sys.argv)
