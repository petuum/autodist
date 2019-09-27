import os
import numpy as np
import tensorflow as tf
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer('oov_bucket_size', 1, "The number of out-of-vocabulary buckets")
flags.DEFINE_integer('batch_size', 128, 'Batch size')
flags.DEFINE_integer('num_steps', 20, 'Number of steps')
flags.DEFINE_float('learning_rate', 0.2, 'Learning rate')
flags.DEFINE_float('max_grad_norm', 10.0, 'max_grad_norm')
flags.DEFINE_integer('num_epoch', 5, 'Number of epoch')


class LM:
    def __init__(self, num_steps):
        self.num_steps = num_steps

        # Use keep_prob 1.0 at evaluation
        self.keep_prob = 0.9
        self.vocab_size = 793470 + FLAGS.oov_bucket_size
        self.emb_size = 512
        self.state_size = 2048
        self.projected_size = 512

        # Use num_sampled 0 (full softmax) at evaluation
        self.num_sampled = 8192
        self.global_step = 0

        self.emb = tf.Variable(tf.random.uniform([self.vocab_size, self.emb_size]),
                               name='emb',
                               trainable=True,
                               dtype=tf.float32)
        self.softmax_w = tf.Variable(tf.random.uniform([self.vocab_size, self.projected_size]),
                                     name='softmax_w',
                                     trainable=True,
                                     dtype=tf.float32)
        self.softmax_b = tf.Variable(tf.zeros([self.vocab_size]),
                                     name='softmax_b',
                                     trainable=True,
                                     dtype=tf.float32)
        self.W = tf.Variable(tf.zeros([self.emb_size + self.projected_size, 4 * self.state_size]),
                             name='W',
                             trainable=True,
                             dtype=tf.float32)
        self.B = tf.Variable(tf.zeros([4 * self.state_size]),
                             name='B',
                             trainable=True,
                             dtype=tf.float32)
        self.W_P = tf.Variable(tf.zeros([self.state_size, self.projected_size]),
                               name='W_P',
                               trainable=True,
                               dtype=tf.float32)
        # self.optimizer = tf.optimizers.Adagrad(FLAGS.learning_rate, initial_accumulator_value=1.0)
        self.optimizer = tf.optimizers.SGD(lr=FLAGS.learning_rate)
        self.c = np.zeros([FLAGS.batch_size, self.state_size], dtype=np.float32)
        self.h = np.zeros([FLAGS.batch_size, self.projected_size], dtype=np.float32)

    def forward(self, x, y, w, training):
        # [bs, steps, emb_size]
        x = tf.compat.v1.nn.embedding_lookup(self.emb, x, partition_strategy='div')
        if training:
            x = tf.nn.dropout(x, 1 - self.keep_prob)

        # [bs, emb_size] * steps
        inputs = [tf.squeeze(v, axis=[1]) for v in tf.split(value=x, num_or_size_splits=self.num_steps, axis=1)]
        for t in range(self.num_steps):
            cell_inputs = tf.concat([inputs[t], self.h], axis=1)
            lstm_matrix = tf.linalg.matmul(cell_inputs, self.W) + self.B
            i, j, f, o = tf.split(lstm_matrix, 4, axis=1)
            self.c = tf.math.sigmoid(f + 1.0) * self.c + tf.math.sigmoid(i) * tf.math.tanh(j)
            self.h = tf.math.sigmoid(o) * tf.math.tanh(self.c)
            self.h = tf.linalg.matmul(self.h, self.W_P)
            inputs[t] = self.h
            if training:
                inputs[t] = tf.nn.dropout(inputs[t], 1 - self.keep_prob)

        inputs[t] = tf.identity(inputs[t])
        inputs = tf.reshape(tf.concat(inputs, axis=1), [-1, self.projected_size])

        if training:
            targets = tf.reshape(y, [-1, 1])
            loss = tf.nn.sampled_softmax_loss(self.softmax_w,
                                              self.softmax_b,
                                              targets,
                                              inputs,
                                              self.num_sampled,
                                              self.vocab_size)
        else:
            full_softmax_w = tf.reshape(tf.concat(self.softmax_w, axis=1), [-1, self.projected_size])
            full_softmax_w = full_softmax_w[:self.vocab_size, :]

            logits = tf.matmul(inputs, full_softmax_w, transpose_b=True) + self.softmax_b
            targets = tf.reshape(y, [-1])
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)

        loss = tf.reduce_mean(loss * tf.reshape(tf.dtypes.cast(w, tf.float32), [-1]))
        return loss

    def train_step(self, input_data):
        # Note(Hao): enforce the lookup and lookup init to be replicated
        # with ops.device('/cpu:0'):
        text_init = tf.lookup.TextFileInitializer(os.path.join(FLAGS.datadir, "1b_word_vocab.txt"), tf.string, 0,
                                                  tf.int64, -1, delimiter=" ")
        self.vocab = tf.lookup.StaticVocabularyTable(text_init, FLAGS.oov_bucket_size)
        xy = tf.dtypes.cast(self.vocab.lookup(input_data), dtype=tf.int32)
        x = xy[:, 0, :]
        y = xy[:, 1, :]
        w = np.ones([FLAGS.batch_size, FLAGS.num_steps])

        # FIXME: cast a PartitionVariabvles to a list of Variables
        # so that we can derive its gradients
        # otherwise will throw an error
        emb_vars = [self.emb]
        lstm_vars = [self.W, self.B, self.W_P]
        softmax_vars = [self.softmax_w, self.softmax_b]
        all_vars = emb_vars + lstm_vars + softmax_vars

        # all_vars = [self.emb, self.softmax_w, self.W, self.B, self.W_P, self.softmax_b]
        loss = self.forward(x, y, w, training=True)
        scaled_loss = loss * FLAGS.num_steps
        grads = tf.gradients(scaled_loss, all_vars)

        # Embedding grads
        emb_grads = grads[:len(emb_vars)]
        emb_grads = [tf.IndexedSlices(grad.values * FLAGS.batch_size,
                                      grad.indices,
                                      grad.dense_shape) for grad in emb_grads]

        lstm_grads = grads[len(emb_vars):len(emb_vars) + len(lstm_vars)]
        lstm_grads, _ = tf.clip_by_global_norm(lstm_grads, FLAGS.max_grad_norm)

        softmax_grads = grads[len(emb_vars) + len(lstm_vars):]

        clipped_grads = emb_grads + lstm_grads + softmax_grads
        grads_and_vars = list(zip(clipped_grads, all_vars))

        train_op = self.optimizer.apply_gradients(grads_and_vars)
        # FIXME(Hao): The code below does not work in TF2.x,
        # ema = tf.train.ExponentialMovingAverage(decay=0.999)
        # with tf.control_dependencies([train_op]):
        # ema.apply([self.softmax_w, self.softmax_b])
        return loss, train_op
