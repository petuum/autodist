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
flags.DEFINE_integer('train_batch_size', 8, 'Total batch size for training.')
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
                            train_batch_size):
    """Run BERT pretrain model training using low-level API."""
    if strategy is not None:
        num_replicas_in_sync = strategy.num_replicas_in_sync
    else:
        num_replicas_in_sync = 1

    train_input_fn = get_pretrain_dataset_fn(input_files, max_seq_length,
                                             max_predictions_per_seq,
                                             train_batch_size,
                                             num_replicas_in_sync)

    def _get_pretrain_model():
        """Gets a pretraining model."""
        pretrain_model, core_model = bert_models.pretrain_model(
            bert_config, max_seq_length, max_predictions_per_seq)

        pretrain_model.optimizer = tf.optimizers.Adam(lr=initial_lr)
        if FLAGS.fp16_implementation == 'graph_rewrite':
            pretrain_model.optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(
                pretrain_model.optimizer)
        return pretrain_model, core_model

    time_callback = keras_utils.TimeHistory(
        train_batch_size * steps_per_loop, 1)

    ##########################################################################
    # Build with Graph mode and AutoDist scope in bert_utils
    trained_model = bert_utils.run_customized_training_loop(
        strategy=strategy,
        model_fn=_get_pretrain_model,
        loss_fn=get_loss_fn(
            loss_factor=1.0 /
            num_replicas_in_sync if FLAGS.scale_loss else 1.0),
        model_dir=model_dir,
        train_input_fn=train_input_fn,
        steps_per_epoch=steps_per_epoch,
        steps_per_loop=steps_per_loop,
        epochs=epochs,
        sub_model_export_name='pretrained/bert_model',
        custom_callbacks=[time_callback])
    ##########################################################################
    return trained_model


def run_bert_pretrain(strategy, gpu_num=1, node_num=1):
    """Runs BERT pre-training."""

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    logging.info(
        'Training using customized training loop TF 2.0 with distrubuted'
        'strategy.')

    return run_customized_training(
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
        FLAGS.train_batch_size * gpu_num * node_num)


def main(_):
    assert tf.version.VERSION.startswith('2.')

    if not FLAGS.model_dir:
        FLAGS.model_dir = '/tmp/bert/'

    #########################################################################
    # Construct AutoDist with ResourceSpec for Different Strategies
    if FLAGS.autodist_patch_tf:
        os.environ['AUTODIST_PATCH_TF'] = 'True'
    else:
        os.environ['AUTODIST_PATCH_TF'] = 'False'
    resource_spec_file = os.path.join(
        os.path.dirname(__file__),
        '../resource_spec.yml')

    if FLAGS.autodist_strategy == 'PS':
        strategy = AutoDist(
            resource_spec_file, PS(
                local_proxy_variable=FLAGS.proxy))
    elif FLAGS.autodist_strategy == 'PSLoadBalancing':
        strategy = AutoDist(
            resource_spec_file, PSLoadBalancing(
                local_proxy_variable=FLAGS.proxy))
    elif FLAGS.autodist_strategy == 'PartitionedPS':
        strategy = AutoDist(
            resource_spec_file, PartitionedPS(
                local_proxy_variable=FLAGS.proxy))
    elif FLAGS.autodist_strategy == 'AllReduce':
        strategy = AutoDist(
            resource_spec_file, AllReduce(
                chunk_size=FLAGS.chunk_size))
    elif FLAGS.autodist_strategy == 'Parallax':
        strategy = AutoDist(
            resource_spec_file,
            Parallax(
                chunk_size=FLAGS.chunk_size,
                local_proxy_variable=FLAGS.proxy))
    else:
        raise ValueError(
            'the strategy can be only from PS, PSLoadBalancing, PartitionedPS, AllReduce, Parallax')

    strategy.num_replicas_in_sync = strategy._resource_spec.num_gpus

    if strategy:
        print('***** Number of cores used : ', strategy.num_replicas_in_sync)

    resource_info = yaml.safe_load(open(resource_spec_file, 'r'))
    try:
        node_num = len(resource_info['nodes'])
    except ValueError:
        print("nodes need to be set in specficiation file")

    try:
        gpu_num = len(resource_info['nodes'][0]['gpus'])
    except ValueError:
        print("gpus need to be set in specficiation file")
    #########################################################################

    logdir = '/tmp/logs'
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    logname = 'bert_strategy_{}_node_{}_gpu_{}_patch_{}_proxy_{}'.format(
        FLAGS.autodist_strategy, node_num, gpu_num, FLAGS.autodist_patch_tf, FLAGS.proxy)

    logging.get_absl_handler().use_absl_log_file(logname, logdir)
    # start running
    run_bert_pretrain(strategy, gpu_num, node_num)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
