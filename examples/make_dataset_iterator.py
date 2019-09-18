import sys
import os
import numpy as np
import tensorflow as tf

from autodist import AutoDist
from tensorflow.python.training.training_util import get_or_create_global_step

resource_spec_file = os.path.join(os.path.dirname(__file__), 'resource_spec.yml')

def prepare_mnist_features_and_labels(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)
    return x, y

def mnist_dataset():
    (x, y), _ = tf.keras.datasets.fashion_mnist.load_data()

    x = x[:, :, :, None]

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(prepare_mnist_features_and_labels)
    ds = ds.take(20000).shuffle(20000).batch(32)
    return ds

def main(_):
    autodist = AutoDist(resource_spec_file, 'PS')
    d = autodist

    EPOCHS = 1
    train_steps_per_epoch = 8

    with d.scope():
        iterator = d.make_dataset_iterator(mnist_dataset)

        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(2048, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        optimizer = tf.keras.optimizers.SGD()

        def train_step(inputs):
            x, y = inputs
            with tf.GradientTape() as tape:
                y_hat = model(x, training=True)
                loss = loss_fn(y, y_hat)
                all_vars = []
                for v in model.trainable_variables:
                    all_vars.append(v)
                grads = tf.gradients(loss, all_vars)
            update = optimizer.apply_gradients(zip(grads, all_vars))

            return loss, update, optimizer.iterations

        for epoch in range(EPOCHS):
            for _ in range(train_steps_per_epoch):
                loss, _, i = d.run(train_step, iterator)
                print(f"step: {i}, train_loss: {loss}")


main(sys.argv)
