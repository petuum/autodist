import os
import numpy as np
import tensorflow as tf
from iterator import MyIterator

############################################################
# Step 1: Construct AutoDist with ResourceSpec
from autodist import AutoDist
filepath = os.path.join(os.path.dirname(__file__), 'resource_spec.yml')
autodist = AutoDist(resource_spec_file=filepath)
############################################################

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images[:512, :, :, None]
test_images = test_images[:512, :, :, None]
train_labels = train_labels[:512]
test_labels = test_labels[:512]
train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)

BATCH_SIZE = 64
EPOCHS = 3

#############################################################
# Step 2: Build with Graph mode, and put it under AutoDist scope
with tf.Graph().as_default(), autodist.scope():
#############################################################

    # train_dataset = tf.data.Dataset.from_tensor_slices(
    #     (train_images, train_labels)).repeat(EPOCHS).shuffle(len(train_images)//2).batch(BATCH_SIZE)

    # train_iterator = tf.compat.v1.data.make_one_shot_iterator(train_dataset).get_next()

    # x = tf.compat.v1.placeholder(tf.float32, (None, 28, 28, 1), name='x') 
    # y = tf.compat.v1.placeholder(tf.int8, (None, ), name='y')

    x = tf.keras.Input(shape=(28, 28, 1), dtype = tf.float32)
    y = tf.keras.Input(shape=(), dtype = tf.int8)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD()

    train_iterator = MyIterator(train_images, train_labels, 2)

    def train_step(x, y):

        y_hat = model(x, training=True)
        loss = loss_fn(y, y_hat)
        all_vars = []
        for v in model.trainable_variables:
            all_vars.append(v)
        grads = tf.gradients(loss, all_vars)
        update = optimizer.apply_gradients(zip(grads, all_vars))

        return loss, update

    fetches = train_step(x,y)
    #############################################################
    # Step 3: create distributed session
    sess = autodist.create_distributed_session()
    #############################################################
    for i in range(min(10, len(train_images) // BATCH_SIZE * EPOCHS)):

        if i%4 == 0:
            train_iterator.batch_size *= 2

        input = next(train_iterator)
        print("current batch_size: {}".format(len(input[1])))
        print("currrent lr: {}".format(sess.run(optimizer.lr)))
        loss, _ = sess.run(fetches, feed_dict = {x: input[0], y: input[1]})
        print(f"train_loss: {loss}")

