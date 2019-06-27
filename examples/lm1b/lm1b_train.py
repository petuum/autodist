import os
import time
import json

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf
from autodist.ps_strategy import PSStrategy

from data_utils import gen_lm1b_train_dataset
import language_model


FLAGS = flags.FLAGS
flags.DEFINE_string("logdir", "/tmp/lm1b", "Logging directory.")
flags.DEFINE_string("datadir", None, "Logging directory.")
flags.DEFINE_integer("eval_steps", 70, "Number of eval steps.")
flags.DEFINE_integer('max_steps', 1000000,
                     """Number of iterations to run for each workers.""")
flags.DEFINE_integer('log_frequency', 100,
                     """How many steps between two runop logs.""")

file = os.path.join(os.path.dirname(__file__), 'cluster_spec.json')
cluster_spec = tf.train.ClusterSpec(json.load(open(file)))

def main(_):
    data_path = os.path.join(FLAGS.datadir, "training-monolingual.tokenized.shuffled/*")
    strategy = PSStrategy(cluster_spec, cost_model=False)
    distribute_batch_size = FLAGS.batch_size * strategy.num_replicas_in_sync
    with strategy.scope():
        train_dataset = gen_lm1b_train_dataset(data_path, FLAGS.num_steps)
        train_dataset = train_dataset.batch(distribute_batch_size)
        train_iterator = strategy.make_dataset_iterator(train_dataset)

        def train_step(inputs):
            model = language_model.LM(FLAGS.num_steps)
            return model.train_step(inputs)

        prev_time = time.time()
        for local_step in range(FLAGS.max_steps):
            loss, _ = strategy.experimental_run(train_step, train_iterator)
            if local_step % FLAGS.log_frequency == 0:
                cur_time = time.time()
                elapsed_time = cur_time - prev_time
                num_words = distribute_batch_size * FLAGS.log_frequency
                wps = float(num_words) / elapsed_time
                logging.info("Iteration %d, time = %.2fs, wps = %.0f, train loss = %.4f" % (
                    local_step, cur_time - prev_time, wps, loss))
                prev_time = cur_time



app.run(main)

