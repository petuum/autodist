# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# It includes the derived work based on:
#
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import yaml
import os
import sys
import ray
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging

from utils.logs import logger
from utils.misc import keras_utils

from utils import bert_modeling as modeling
from utils import bert_models
from utils import common_flags
from utils import input_pipeline
from utils import bert_utils
from utils import ray_utils

#########################################################################
# Import AutoDist and Strategy
from autodist import AutoDist
from autodist.strategy.all_reduce_strategy import AllReduce
from autodist.strategy.ps_strategy import PS
from autodist.strategy.ps_lb_strategy import PSLoadBalancing
from autodist.strategy.parallax_strategy import Parallax
from autodist.strategy.partitioned_ps_strategy import PartitionedPS
#########################################################################

flags.DEFINE_string(
    'input_files',
    None,
    'File path to retrieve training data for pre-training.')
flags.DEFINE_integer(
    'max_seq_length', 128,
    'The maximum total input sequence length after WordPiece tokenization. '
    'Sequences longer than this will be truncated, and sequences shorter '
    'than this will be padded.')
flags.DEFINE_integer('max_predictions_per_seq', 20,
                     'Maximum predictions per sequence_output.')
flags.DEFINE_integer('train_batch_size', 2, 'Total batch size for training.')
flags.DEFINE_integer('chunk_size', 256, 'The chunk size for training.')
flags.DEFINE_integer('num_steps_per_epoch', 1000,
                     'Total number of training steps to run per epoch.')
flags.DEFINE_string(
    name='autodist_strategy',
    default='PS',
    help='the autodist strategy')
flags.DEFINE_boolean(
    name='autodist_patch_tf',
    default=True,
    help='AUTODIST_PATCH_TF')
flags.DEFINE_string(
    'address',
    'auto',
    'IP address of the Ray head node')

flags.DEFINE_boolean(name='proxy', default=True, help='turn on off the proxy')


common_flags.define_common_bert_flags()

FLAGS = flags.FLAGS


def get_pretrain_dataset_fn(input_file_pattern, seq_length,
                            max_predictions_per_seq, global_batch_size,
                            num_replicas_in_sync):
    """Returns input dataset from input file string."""
    def _dataset_fn(ctx=None):
        """Returns tf.data.Dataset for distributed BERT pretraining."""
        input_patterns = input_file_pattern.split(',')
        batch_size = int(global_batch_size / num_replicas_in_sync)
        train_dataset = input_pipeline.create_pretrain_dataset(
            input_patterns,
            seq_length,
            max_predictions_per_seq,
            batch_size,
            is_training=True)
        return train_dataset

    return _dataset_fn


def get_loss_fn(loss_factor=1.0):
    """Returns loss function for BERT pretraining."""

    def _bert_pretrain_loss_fn(unused_labels, losses, **unused_args):
        return tf.keras.backend.mean(losses) * loss_factor

    return _bert_pretrain_loss_fn


def run_customized_training(strategy,
                            bert_config,
                            max_seq_length,
                            max_predictions_per_seq,
                            model_dir,
                            steps_per_epoch,
                            steps_per_loop,
                            epochs,
                            initial_lr,
                            input_files,
                            train_batch_size,
                            num_replicas):
    def _get_pretrain_model():
        """Gets a pretraining model."""
        pretrain_model, core_model = bert_models.pretrain_model(
            bert_config, max_seq_length, max_predictions_per_seq)

        pretrain_model.optimizer = tf.optimizers.Adam(lr=initial_lr)
        return pretrain_model, core_model

    time_callback = keras_utils.TimeHistory(
        train_batch_size * steps_per_loop, 1)

    train_input_fn = get_pretrain_dataset_fn(input_files, max_seq_length,
                                             max_predictions_per_seq,
                                             train_batch_size,
                                             num_replicas)

    ray_utils.run_ray_job(strategy=strategy,
                          model_fn=_get_pretrain_model,
                          loss_fn=get_loss_fn(loss_factor=1.0),
                          model_dir=model_dir,
                          train_input_fn=train_input_fn,
                          steps_per_epoch=steps_per_epoch,
                          steps_per_loop=steps_per_loop,
                          epochs=epochs,
                          sub_model_export_name='pretrained/bert_model',
                          custom_callbacks=[time_callback])


def run_bert_pretrain(strategy, num_gpus=1, num_nodes=1):
    """Runs BERT pre-training."""

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    logging.info(
        'Training using customized training loop TF 2.0 with AutoDist')

    run_customized_training(
        strategy,
        bert_config,
        FLAGS.max_seq_length,
        FLAGS.max_predictions_per_seq,
        FLAGS.model_dir,
        FLAGS.num_steps_per_epoch,
        FLAGS.steps_per_loop,
        FLAGS.num_train_epochs,
        FLAGS.learning_rate,
        FLAGS.input_files,
        FLAGS.train_batch_size * num_nodes * num_gpus,
        num_nodes * num_gpus)


def main(_):
    assert tf.version.VERSION.startswith('2.')

    if not FLAGS.model_dir:
        FLAGS.model_dir = "/tmp/ckpt/"

    #########################################################################
    # Construct AutoDist with ResourceSpec for Different Strategies
    if FLAGS.autodist_patch_tf:
        os.environ['AUTODIST_PATCH_TF'] = 'True'
    else:
        os.environ['AUTODIST_PATCH_TF'] = 'False'

    strategy_table = {'PS': PS(local_proxy_variable=FLAGS.proxy),
                      'PSLoadBalancing': PSLoadBalancing(local_proxy_variable=FLAGS.proxy),
                      'PartitionedPS': PartitionedPS(local_proxy_variable=FLAGS.proxy),
                      'AllReduce': AllReduce(chunk_size=FLAGS.chunk_size),
                      'Parallax': Parallax(chunk_size=FLAGS.chunk_size,
                                           local_proxy_variable=FLAGS.proxy)}

    if FLAGS.autodist_strategy not in strategy_table: 
        raise ValueError(
                f"the strategy can be only from {','.join(strategy_table.keys())}")

    logdir = '/tmp/logs'
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    ray.init(address=FLAGS.address)
    num_nodes = len(ray.nodes())
    num_gpus_per_node = max(1, ray.nodes()[0]['Resources'].get('GPU', 0))
    
    logname = 'bert_strategy_{}_node_{}_gpu_{}_patch_{}_proxy_{}'.format(
        FLAGS.autodist_strategy, num_nodes, num_gpus_per_node, FLAGS.autodist_patch_tf, FLAGS.proxy)

    logging.get_absl_handler().use_absl_log_file(logname, logdir)

    run_bert_pretrain(strategy_table[FLAGS.autodist_strategy], num_gpus_per_node, num_nodes)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
