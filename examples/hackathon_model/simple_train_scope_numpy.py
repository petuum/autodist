import os
import time
import tensorflow as tf
import numpy as np
from absl import app

from autodist import AutoDist

resource_spec_file = os.path.join(os.path.dirname(__file__), '../resource_spec.yml')
autodist = AutoDist(resource_spec_file, 'PS')

vocab_size = 10000
embedding_size = 16
hidden_dim = 16
max_steps = 1000
batch_size = 128
log_frequency = 100


class SimpleModel():
    def __init__(self):
        self.emb = tf.Variable(tf.random.uniform([vocab_size, embedding_size]),
                               name='emb',
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
        self.optimizer = tf.optimizers.SGD(lr=0.0005)

    def forward(self, x, y):
        # embedding layer
        x = tf.nn.embedding_lookup(self.emb, x)
        # global average pool
        x = tf.math.reduce_mean(x, axis=1)
        # dense
        x = tf.linalg.matmul(x, self.w1) + self.b1
        x = tf.nn.relu(x)
        logits = tf.linalg.matmul(x, self.w2) + self.b2  # logits
        logits = tf.squeeze(logits)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(loss)
        return loss

    @autodist.function
    def train_fn(self, x, y):
        loss = self.forward(x, y)
        trainables = [self.emb, self.w1, self.b1, self.w2, self.b2]
        gradients = tf.gradients(loss, trainables)
        #gradients = tape.gradient(loss, trainables)
        # strategy requires users to provide the train_op handle
        train_op = self.optimizer.apply_gradients(zip(gradients, trainables))
        return loss, train_op


def main(_):
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
    left = 0
    with autodist.scope():  # AutoDist code
        model = SimpleModel()
        prev_time = time.time()
        print(train_data.shape, test_data.shape)
        for local_step in range(max_steps):
            x = train_data[left:left + batch_size]
            y = train_labels[left:left + batch_size]
            # fetch train_op and loss
            loss, _ = model.train_fn(x=x, y=y)
            if local_step % log_frequency == 0:
                cur_time = time.time()
                elapsed_time = cur_time - prev_time
                num_sentences = batch_size * log_frequency
                wps = float(num_sentences) / elapsed_time
                print("Iteration %d, time = %.2fs, wps = %.0f, train loss = %.4f" % (
                    local_step, cur_time - prev_time, wps, loss))
                prev_time = cur_time
            left = left + batch_size
            if (left + batch_size) > train_data.shape[0]:
                break

app.run(main)