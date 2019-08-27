import os
import time
import json
import glob

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
from autodist import AutoDist

import language_model

FLAGS = flags.FLAGS
flags.DEFINE_string("logdir", "/tmp/lm1b", "Logging directory.")
flags.DEFINE_string("datadir", "/tmp/dataset/lm1b", "Data directory.")
flags.DEFINE_integer("eval_steps", 70, "Number of eval steps.")
flags.DEFINE_integer('max_steps', 1000000,
                     """Number of iterations to run for each workers.""")
flags.DEFINE_integer('log_frequency', 100,
                     """How many steps between two runop logs.""")

resource_spec_file = os.path.join(os.path.dirname(__file__), '../resource_spec.yml')
config_file = os.path.join(os.path.dirname(__file__), '../runner_config.yml')
# autodist = AutoDist(resource_spec_file, 'PS')
autodist = AutoDist(resource_spec_file, 'PS', config_file)


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


def main(_):
    data_path = os.path.join(FLAGS.datadir, "training-monolingual.tokenized.shuffled/*")
    distribute_batch_size = FLAGS.batch_size * autodist._resource_spec.num_gpus
    with autodist.scope():
        train_dataset = gen_lm1b_train_dataset(data_path, FLAGS.num_steps)
        train_dataset = train_dataset.batch(FLAGS.batch_size)
        train_iterator = train_dataset.make_one_shot_iterator().get_next()

        model = language_model.LM(FLAGS.num_steps)
        # TODO (Hao): need to improve this.
        train_step = autodist.function(model.train_step)

        prev_time = time.time()
        for local_step in range(FLAGS.max_steps):
            loss, _ = train_step(train_iterator)
            if local_step % FLAGS.log_frequency == 0:
                cur_time = time.time()
                elapsed_time = cur_time - prev_time
                num_words = distribute_batch_size * FLAGS.log_frequency
                wps = float(num_words) / elapsed_time
                logging.info("Iteration %d, time = %.2fs, wps = %.0f, train loss = %.4f" % (
                    local_step, cur_time - prev_time, wps, loss))
                prev_time = cur_time


app.run(main)
