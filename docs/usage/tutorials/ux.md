# Build differently on TensorFlow


### Support for Keras Model
```python
import tensorflow as tf
from autodist import AutoDist
d = AutoDist(resource_spec_file='resource_spec.yml')

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
```

### Support for Decorator (Experimental)

Use `@autodist.function` in TensorFlow graph mode as `@tf.function`.

```python
import time
import tensorflow as tf
import numpy as np


def main(autodist):
    vocab_size = 10000
    embedding_size = 16
    hidden_dim = 16
    max_steps = 200
    batch_size = 128
    log_frequency = 100

    class SimpleModel:
        def __init__(self):
            self.emb1 = tf.Variable(tf.random.uniform([vocab_size, embedding_size]),
                                   name='emb1',
                                   trainable=True,
                                   dtype=tf.float32)
            self.emb2 = tf.Variable(tf.random.uniform([vocab_size, embedding_size]),
                                   name='emb2',
                                   trainable=True,
                                   dtype=tf.float32)
            self.w1 = tf.Variable(tf.random.uniform([embedding_size, hidden_dim]),
                                  name='w1',
                                  trainable=True,
                                  dtype=tf.float32)
            self.b1 = tf.Variable(tf.zeros([hidden_dim]),
                                  name='b1',
                                  trainable=True,
                                  dtype=tf.float32)
            self.w2 = tf.Variable(tf.random.uniform([hidden_dim, 1]),
                                  name='w2',
                                  trainable=True,
                                  dtype=tf.float32)
            self.b2 = tf.Variable(tf.zeros([1]),
                                  name='b2',
                                  trainable=True,
                                  dtype=tf.float32)
            major_version, _, _ = tf.version.VERSION.split('.')
            if major_version == '1':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=0.2)
            else:
                self.optimizer = tf.optimizers.Adam(learning_rate=0.2)

        def forward(self, x, y):
            x, z = tf.nn.embedding_lookup(self.emb1, x), tf.nn.embedding_lookup(self.emb2, x)
            z = tf.cond(tf.reduce_sum(z) > 0, lambda: tf.identity(z), lambda: tf.identity(z))
            x = .5 * (x + z)
            x = tf.math.reduce_mean(x, axis=1)
            x = tf.linalg.matmul(x, self.w1) + self.b1
            x = tf.nn.relu(x)
            logits = tf.linalg.matmul(x, self.w2) + self.b2  # logits
            logits = tf.squeeze(logits)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
            loss = tf.reduce_mean(loss)
            return loss

        @autodist.function
        def train_fn(self, xy):
            x, y = xy
            loss = self.forward(x, y)
            trainables = [self.emb1, self.emb2, self.w1, self.b1, self.w2, self.b2]
            gradients = tf.gradients(loss, trainables)
            train_op = self.optimizer.apply_gradients(zip(gradients, trainables))
            return loss, train_op, self.emb1, self.emb2

    (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)
    train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data,
                                                               value=0,
                                                               padding='post',
                                                               maxlen=256)
    test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data,
                                                              value=0,
                                                              padding='post',
                                                              maxlen=256)
    train_labels = train_labels.astype(np.float32)
    with tf.Graph().as_default(), autodist.scope():  # AutoDist code
        my_iterator = tf.compat.v1.data.Dataset.from_tensor_slices((train_data, train_labels)) \
            .shuffle(25000).batch(batch_size).repeat().make_one_shot_iterator().get_next()
        model = SimpleModel()
        prev_time = time.time()
        for local_step in range(max_steps):
            # fetch train_op and loss
            loss, _, _, _ = model.train_fn(my_iterator)
            if local_step % log_frequency == 0:
                cur_time = time.time()
                elapsed_time = cur_time - prev_time
                num_sentences = batch_size * log_frequency
                wps = float(num_sentences) / elapsed_time
                print("Iteration %d, time = %.2fs, wps = %.0f, train loss = %.4f" % (
                    local_step, cur_time - prev_time, wps, loss))
                prev_time = cur_time
```