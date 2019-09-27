import numpy as np
import os
import tensorflow as tf

############################################################
# Change 1: Construct AutoDist with ResourceSpec
from autodist import AutoDist
resource_spec_file = os.path.join(os.path.dirname(__file__), 'resource_spec.yml')
d = AutoDist(resource_spec_file, 'PSLoadBalancing')
#############################################################

NUM_DATAPOINTS = 100

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (_, _) = fashion_mnist.load_data()
train_images = train_images[:, :, :, None]
train_images = train_images / np.float32(255)

# Trim dataset for smaller graphdef size
train_images = train_images[0:NUM_DATAPOINTS, :, :, :]
train_labels = train_labels[0:NUM_DATAPOINTS]

BATCH_SIZE = 32
STEPS_PER_EPOCH = len(train_images) // BATCH_SIZE
EPOCHS = 100

#############################################################
# Change 2: Put Model under the Scope
with d.scope():
#############################################################

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_images, train_labels)).repeat(EPOCHS).shuffle(
        NUM_DATAPOINTS).batch(BATCH_SIZE)

    #############################################################
    # Change 3.1: Construct Graph-Mode Iterator
    # train_iterator = iter(train_dataset)  # original code
    train_iterator = tf.compat.v1.data.make_one_shot_iterator(train_dataset)
    batch = train_iterator.get_next()
    #############################################################

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

    #############################################################
    # Change 4: Mark the Training Step
    @d.function
    #############################################################
    def train_step(inputs):
        x, y = inputs
        with tf.GradientTape() as tape:
            y_hat = model(x, training=True)
            loss = loss_fn(y, y_hat)
            # grads = tape.gradient(loss, model.trainable_variables)
            grads = tf.gradients(loss, model.trainable_variables)
            #############################################################
            # Change 5: Return the Training Op
            # optimizer.apply_gradients(zip(grads, model.trainable_variables))  # original code
            train_op = optimizer.apply_gradients(zip(grads, model.trainable_variables))
            #############################################################
        return optimizer.iterations, loss, train_op

    for epoch in range(EPOCHS):
        for _ in range(STEPS_PER_EPOCH):
            #############################################################
            # Change 3.2: Use the Graph-Mode Iterator
            # batch = next(train_iterator)  # original code
            #############################################################
            i, loss, _ = train_step(batch)
            print("step: {}, train_loss: {:5f}".format(int(i), loss))
