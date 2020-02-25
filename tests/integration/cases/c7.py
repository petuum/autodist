import tensorflow as tf

def main(autodist):

    d = autodist

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    with tf.Graph().as_default(), d.scope():
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10)
        ])
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer='adam',
                      loss=loss_fn,
                      metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=1)
        print(model.evaluate(x_test,  y_test, verbose=2))
