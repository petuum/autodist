
import numpy as np
import tensorflow as tf


def main(autodist):

    d = autodist

    fashion_mnist = tf.keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images[:512, :, :, None]
    test_images = test_images[:512, :, :, None]
    train_labels = train_labels[:512]
    test_labels = test_labels[:512]
    print(train_images.shape, train_labels.shape)

    train_images = train_images / np.float32(255)
    test_images = test_images / np.float32(255)

    BUFFER_SIZE = len(train_images)

    BATCH_SIZE = 32

    EPOCHS = 1
    train_steps_per_epoch = 8

    with tf.Graph().as_default(), d.scope():

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (train_images, train_labels)).shuffle(
            BUFFER_SIZE).batch(BATCH_SIZE)

        train_iterator = tf.compat.v1.data.make_one_shot_iterator(train_dataset).get_next()

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
        def train_step(inputs):
            x, y = inputs
            with tf.GradientTape() as tape:
                y_hat = model(x, training=True)
                loss = loss_fn(y, y_hat)
                all_vars = []
                for v in model.trainable_variables:
                    all_vars.append(v)
                # grads = tape.gradient(loss, all_vars)
                # grads = tf.gradients(loss, all_vars)
                grads = optimizer.get_gradients(loss, all_vars)
            update = optimizer.apply_gradients(zip(grads, all_vars))

            return loss, update

        for epoch in range(EPOCHS):
            for _ in range(train_steps_per_epoch):
                loss, _ = train_step(train_iterator)
                print(f"train_loss: {loss}")
