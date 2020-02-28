import numpy as np
import tensorflow as tf

def main(autodist):

    NUM_DATAPOINTS = 100

    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (_, _) = fashion_mnist.load_data()
    train_images = train_images[:, :, :, None]
    train_images = train_images / np.float32(255)

    train_images = train_images[0:NUM_DATAPOINTS, :, :, :]
    train_labels = train_labels[0:NUM_DATAPOINTS]

    BATCH_SIZE = 32
    STEPS_PER_EPOCH = len(train_images) // BATCH_SIZE
    EPOCHS = 5



    with tf.Graph().as_default(), autodist.scope():
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (train_images, train_labels)).repeat(EPOCHS).shuffle(
            NUM_DATAPOINTS).batch(BATCH_SIZE)


        train_iterator = tf.compat.v1.data.make_one_shot_iterator(train_dataset)
        batch = train_iterator.get_next()

        def model_fn():
            img_input = tf.keras.layers.Input(shape=(28,28,1), batch_size=16)
            x=tf.keras.layers.Conv2D(32, 3, activation='relu')(img_input)
            x=tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
            x=tf.keras.layers.MaxPooling2D()(x)
            x=tf.keras.layers.Flatten()(x)
            #print(x)
            x=tf.keras.layers.Dense(128, activation='relu')(x)
            xw = tf.keras.layers.Dense(1280, activation='relu')(x)
            xw = tf.reduce_sum(xw, 0)
            xw = tf.reshape(xw, [128,10])
            print(xw)
            x = tf.matmul(x,xw)
            print(x)
            x = tf.nn.softmax(x)
            x=tf.keras.layers.Dense(10, activation='softmax')(x)
            return tf.keras.Model(img_input, x, name='network')

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

        x, y = batch
        model = model_fn()
        y_hat = model(x, training=True)
        loss = loss_fn(y, y_hat)
        grads = tf.gradients(loss, model.trainable_variables)
        train_op = optimizer.apply_gradients(zip(grads, model.trainable_variables))
        sess = autodist.create_distributed_session()

        for epoch in range(EPOCHS):
            for _ in range(STEPS_PER_EPOCH):
                iv, lossv, _ = sess.run([optimizer.iterations, loss, train_op])
                print("step: {}, train_loss: {:5f}".format(int(iv), lossv))
