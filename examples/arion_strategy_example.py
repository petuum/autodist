import sys
import os
import numpy as np
import json
import tensorflow as tf
from autodist.ps_strategy import PSStrategy
from tensorflow.python.ops.variables import PartitionedVariable

file = os.path.join(os.path.dirname(__file__), 'cluster_spec.json')
cluster_spec = tf.train.ClusterSpec(json.load(open(file)))


def main(_):
    fashion_mnist = tf.keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images[:512, :, :, None]
    test_images = test_images[:512, :, :, None]
    train_labels = train_labels[:512]
    test_labels = test_labels[:512]
    print(train_images.shape, train_labels.shape)

    train_images = train_images / np.float32(255)
    test_images = test_images / np.float32(255)

    strategies = (PSStrategy(cluster_spec, cost_model=True),
                  tf.distribute.MirroredStrategy())

    d = strategies[0]

    BUFFER_SIZE = len(train_images)

    BATCH_SIZE_PER_REPLICA = 32
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * d.num_replicas_in_sync

    EPOCHS = 1
    train_steps_per_epoch = 8

    with d.scope():
        train_dataset = tf.data.Dataset.from_tensor_slices(
                (train_images, train_labels)).shuffle(
                        BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)

        train_iterator = d.make_dataset_iterator(train_dataset)

        test_dataset = tf.data.Dataset.from_tensor_slices(
            (test_images, test_labels)).batch(GLOBAL_BATCH_SIZE)

        test_iterator = d.make_dataset_iterator(test_dataset)

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam()

        def train_step(inputs):
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(2048, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')
            ])
            x, y = inputs
            with tf.GradientTape() as tape:
                y_hat = model(x, training=True)
                loss = loss_fn(y, y_hat)
            all_vars = []
            for v in model.trainable_variables:
                if isinstance(v, PartitionedVariable):
                    all_vars += list(v)
                else:
                    all_vars.append(v)
            grads = tape.gradient(loss, all_vars)
            update = optimizer.apply_gradients(zip(grads, all_vars))

            return loss, update

        for epoch in range(EPOCHS):
            for _ in range(train_steps_per_epoch):
                loss, _ = d.experimental_run(train_step, train_iterator)
                print(f"train_loss: {loss}")

main(sys.argv)
