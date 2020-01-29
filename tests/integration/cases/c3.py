import sys

import numpy as np
import os
import tensorflow as tf

from autodist import AutoDist
from tensorflow.python.training.training_util import get_or_create_global_step

def main(autodist):

    d = autodist

    fashion_mnist = tf.keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images[:, :, :, None]
    test_images = test_images[:, :, :, None]

    train_images = train_images / np.float32(255)
    test_images = test_images / np.float32(255)

    BATCH_SIZE = 128

    EPOCHS = 1
    train_steps_per_epoch = min(100, len(train_images) // BATCH_SIZE)

    with tf.Graph().as_default(), d.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        optimizer = tf.keras.optimizers.SGD()

        @d.function
        def train_step(x, y):
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
            j = 0
            for _ in range(train_steps_per_epoch):
                loss, _, i = train_step(train_images[j:j+BATCH_SIZE], y=train_labels[j:j+BATCH_SIZE])
                print(f"step: {i}, train_loss: {loss}")
                j += BATCH_SIZE

