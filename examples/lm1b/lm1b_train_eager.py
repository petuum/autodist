import os
import time
import json

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import glob
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

import language_model


def gen_lm1b_train_dataset(file_pattern, num_step):
    """
    Returns: The training dataset (tf.data.Dataset) that has been repeated
    and shuffled
    """
    file_names = []
    for file_name in glob.glob(file_pattern):
        file_names.append(file_name)
    if not file_names:
        raise ValueError
    # create dataset ops
    BUFFER_SIZE = 100000

    # TODO(Hao): have to use v1 APIs
    d = tf.compat.v1.data.TextLineDataset(file_names) \
        .map(lambda string: tf.strings.split([string]).values) \
        .flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x)) \
        .window(num_step, 1, 1, True) \
        .flat_map(lambda x: x.batch(num_step)) \
        .window(2, 1, 1, True) \
        .flat_map(lambda x: x.batch(2)) \
        .shuffle(BUFFER_SIZE, reshuffle_each_iteration=True) \
        .repeat()
    return d


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

def main(argv):
    data_path = os.path.join(FLAGS.datadir, "training-monolingual.tokenized.shuffled/*")

    train_dataset = gen_lm1b_train_dataset(data_path, FLAGS.num_steps)
    train_dataset = train_dataset.batch(FLAGS.batch_size)
    model = language_model.LM(FLAGS.num_steps)
    prev_time = time.time()
    for local_step, input_data in enumerate(train_dataset.take(10)):
        loss, _ = model.train_step(input_data)
        if local_step % FLAGS.log_frequency == 0:
            cur_time = time.time()
            elapsed_time = cur_time - prev_time
            num_words = FLAGS.batch_size * FLAGS.log_frequency
            wps = float(num_words) / elapsed_time
            logging.info("Iteration %d, time = %.2fs, wps = %.0f, train loss = %.4f" % (
                local_step, cur_time - prev_time, wps, loss))
            prev_time = cur_time



app.run(main)

