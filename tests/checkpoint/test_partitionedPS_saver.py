import atexit
import os
import time
import tensorflow as tf
import numpy as np
from multiprocessing import Process

from tensorflow.core.protobuf import config_pb2

from autodist import AutoDist
from autodist.checkpoint.saver import Saver as autodist_saver
from autodist.strategy import PartitionedPS


vocab_size = 10000
embedding_size = 16
hidden_dim = 16
max_steps = 101
batch_size = 128
log_frequency = 100

checkpoint_dir = '/tmp/ps_ckpt/'


def clean_up(fn):
    def run():
        try:
            atexit._clear()
            fn()
        except Exception:
            raise
        finally:
            atexit._run_exitfuncs()
    return run


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
        major_version, _, _ = tf.version.VERSION.split('.')
        if major_version == '1':
            # self.optimizer = tf.train.AdagradOptimizer(learning_rate=0.2)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.2)
        else:
            # self.optimizer = tf.optimizers.Adagrad(learning_rate=0.2, initial_accumulator_value=1.0)
            self.optimizer = tf.optimizers.Adam(learning_rate=0.2)

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

    def train_fn(self, xy):
        x, y = xy
        #with tf.GradientTape() as tape:
        loss = self.forward(x, y)
        trainables = [self.emb, self.w1, self.b1, self.w2, self.b2]
        gradients = tf.gradients(loss, trainables)
        #gradients = tape.gradient(loss, trainables)
        # strategy requires users to provide the train_op handle
        train_op = self.optimizer.apply_gradients(zip(gradients, trainables))
        return loss, train_op, gradients


@clean_up
def train_and_save():
    resource_spec_file = os.path.join(os.path.dirname(__file__), 'resource_spec.yml')
    autodist = AutoDist(resource_spec_file, PartitionedPS())
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
        # my_iterator = MyIterator().get_next()
        model = SimpleModel()
        prev_time = time.time()
        # fetch train_op and loss
        loss_fn, train_op, gradients = model.train_fn(my_iterator)

        saver = autodist_saver()

        sess = autodist.create_distributed_session()
        for local_step in range(max_steps):
            loss, _ = sess.run(fetches=[loss_fn, train_op],
                               options=config_pb2.RunOptions(trace_level=config_pb2.RunOptions.NO_TRACE))
            if local_step % log_frequency == 0:
                cur_time = time.time()
                elapsed_time = cur_time - prev_time
                num_sentences = batch_size * log_frequency
                wps = float(num_sentences) / elapsed_time
                print("Iteration %d, time = %.2fs, wps = %.0f, train loss = %.4f" % (
                    local_step, cur_time - prev_time, wps, loss))
                prev_time = cur_time

        saver.save(sess, checkpoint_dir + "sentiment", global_step=local_step)
        print(sess.run(model.emb))
        print(sess.run(gradients[0]))
    print('ending...')


@clean_up
def fine_tune():
    print("=====================================")
    print("reload test")
    print("=====================================")
    with tf.compat.v1.Session() as sess:
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
        my_iterator = tf.compat.v1.data.Dataset.from_tensor_slices((train_data, train_labels)) \
            .shuffle(25000).batch(batch_size).repeat().make_one_shot_iterator().get_next()
        # my_iterator = MyIterator().get_next()
        model = SimpleModel()
        prev_time = time.time()
        # fetch train_op and loss
        loss_fn, train_op, gradients = model.train_fn(my_iterator)

        tf_saver = tf.compat.v1.train.Saver()
        tf_saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

        for local_step in range(max_steps):
            loss, _ = sess.run(fetches=[loss_fn, train_op],
                               options=config_pb2.RunOptions(trace_level=config_pb2.RunOptions.NO_TRACE))
            if local_step % log_frequency == 0:
                cur_time = time.time()
                elapsed_time = cur_time - prev_time
                num_sentences = batch_size * log_frequency
                wps = float(num_sentences) / elapsed_time
                print("Iteration %d, time = %.2fs, wps = %.0f, train loss = %.4f" % (
                    local_step, cur_time - prev_time, wps, loss))
                prev_time = cur_time
        print(sess.run(model.emb))
        print(sess.run(gradients[0]))
        print('ending...')


def test_partitionedPS_saver():
    p = Process(target=train_and_save)
    p.start()
    p.join()
    if p.exitcode != 0:
        raise SystemExit(f"FAILED running train and save test")

    p = Process(target=fine_tune)
    p.start()
    p.join()
    if p.exitcode != 0:
        raise SystemExit(f"FAILED running restore test")
