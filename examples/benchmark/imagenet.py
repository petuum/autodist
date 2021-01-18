# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# It includes the derived work based on:
#
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Runs a ResNet model on the ImageNet dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
import os
import sys
import time
import yaml
from utils.flags import core as flags_core
from utils.logs import logger
from utils.misc import keras_utils
from utils.misc import model_helpers

from utils import common
from utils import imagenet_preprocessing

#########################################################################
# Import AutoDist and Strategy
from autodist import AutoDist
from autodist.strategy.ps_strategy import PS
from autodist.strategy.ps_lb_strategy import PSLoadBalancing
from autodist.strategy.partitioned_ps_strategy import PartitionedPS
from autodist.strategy.all_reduce_strategy import AllReduce
from autodist.strategy.parallax_strategy import Parallax
#########################################################################

flags.DEFINE_string(
    name='cnn_model',
    default='resnet101',
    help='model to test')
flags.DEFINE_string(
    name='autodist_strategy',
    default='PS',
    help='the autodist strategy')
flags.DEFINE_boolean(
    name='autodist_patch_tf',
    default=True,
    help='AUTODIST_PATCH_TF')
flags.DEFINE_boolean(name='proxy', default=True, help='proxy')


#########################################################################
# Construct AutoDist with ResourceSpec for Different Strategies
resource_spec_file = os.path.join(
    os.path.dirname(__file__),
    '../resource_spec.yml')
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

class TimeHistory(object):
    def __init__(self, batch_size, log_steps):
        self.batch_size = batch_size
        self.log_steps = log_steps
        self.global_steps = 0
        self.epoch_num = 0
        self.examples_per_second = 0
        logging.info("batch steps: %f", log_steps)

    def on_train_end(self):
        self.train_finish_time = time.time()
        elapsed_time = self.train_finish_time - self.train_start_time
        logging.info(
            "total time take: %f,"
            "averaged examples_per_second: %f",
            elapsed_time, self.examples_per_second / self.epoch_num)

    def on_epoch_begin(self, epoch):
        self.epoch_num += 1
        self.epoch_start = time.time()

    def on_batch_begin(self, batch):
        self.global_steps += 1
        if self.global_steps == 1:
            self.train_start_time = time.time()
            self.start_time = time.time()

    def on_batch_end(self, batch, loss):
        """Records elapse time of the batch and calculates examples per second."""
        logging.info(
            "global step:%d, loss value: %f",
            self.global_steps, loss)
        if self.global_steps % self.log_steps == 0:
            timestamp = time.time()
            elapsed_time = timestamp - self.start_time
            examples_per_second = (
                self.batch_size * self.log_steps) / elapsed_time
            logging.info(
                "global step:%d, time_taken: %f,"
                "examples_per_second: %f",
                self.global_steps, elapsed_time, examples_per_second)
            self.examples_per_second += examples_per_second
            self.start_time = timestamp

    def on_epoch_end(self, epoch):
        epoch_run_time = time.time() - self.epoch_start
        logging.info(
            "epoch':%d, 'time_taken': %f",
            epoch, epoch_run_time)


def run(flags_obj):
    """
    Run ResNet ImageNet training and eval loop using native Keras APIs.
    Raises:
        ValueError: If fp16 is passed as it is not currently supported.
    Returns:
        Dictionary of training and eval stats.
    """

    #########################################################################
    # Construct AutoDist with ResourceSpec for Different Strategies
    if flags_obj.autodist_patch_tf:
        os.environ['AUTODIST_PATCH_TF'] = '1'
    else:
        os.environ['AUTODIST_PATCH_TF'] = '0'

    if flags_obj.cnn_model == 'vgg16':
        chunk = 25
    elif flags_obj.cnn_model == 'resnet101':
        chunk = 200
    elif flags_obj.cnn_model == 'inceptionv3':
        chunk = 30
    else:
        chunk = 512

    if flags_obj.autodist_strategy == 'PS':
        autodist = AutoDist(
            resource_spec_file, PS(
                local_proxy_variable=flags_obj.proxy))
    elif flags_obj.autodist_strategy == 'PSLoadBalancing':
        autodist = AutoDist(
            resource_spec_file, PSLoadBalancing(
                local_proxy_variable=flags_obj.proxy))
    elif flags_obj.autodist_strategy == 'PartitionedPS':
        autodist = AutoDist(
            resource_spec_file, PartitionedPS(
                local_proxy_variable=flags_obj.proxy))
    elif flags_obj.autodist_strategy == 'AllReduce':
        autodist = AutoDist(resource_spec_file, AllReduce(chunk_size=chunk))
    elif flags_obj.autodist_strategy == 'Parallax':
        autodist = AutoDist(
            resource_spec_file,
            Parallax(
                chunk_size=chunk,
                local_proxy_variable=flags_obj.proxy))
    else:
        raise ValueError(
            'the strategy can be only from PS, PSLoadBalancing, PartitionedPS, AllReduce, Parallax')
    #########################################################################

    dtype = flags_core.get_tf_dtype(flags_obj)
    if dtype == tf.float16:
        loss_scale = flags_core.get_loss_scale(flags_obj, default_for_fp16=128)
        policy = tf.compat.v1.keras.mixed_precision.experimental.Policy(
            'mixed_float16', loss_scale=loss_scale)
        tf.compat.v1.keras.mixed_precision.experimental.set_policy(policy)
        if not keras_utils.is_v2_0():
            raise ValueError('--dtype=fp16 is not supported in TensorFlow 1.')
    elif dtype == tf.bfloat16:
        policy = tf.compat.v1.keras.mixed_precision.experimental.Policy(
            'mixed_bfloat16')
        tf.compat.v1.keras.mixed_precision.experimental.set_policy(policy)

    input_fn = imagenet_preprocessing.input_fn

    drop_remainder = flags_obj.enable_xla

    if 'vgg' in flags_obj.cnn_model:
        lr_schedule = 0.01
    else:
        lr_schedule = 0.1
    if flags_obj.use_tensor_lr:
        lr_schedule = common.PiecewiseConstantDecayWithWarmup(
            batch_size=flags_obj.batch_size,
            epoch_size=imagenet_preprocessing.NUM_IMAGES['train'],
            warmup_epochs=common.LR_SCHEDULE[0][1],
            boundaries=list(p[1] for p in common.LR_SCHEDULE[1:]),
            multipliers=list(p[0] for p in common.LR_SCHEDULE),
            compute_lr_on_cpu=True)

    #########################################################################
    # Build with Graph mode, and put all under AutoDist scope.
    with tf.Graph().as_default(), autodist.scope():
    ##########################################################################        
        train_input_dataset = input_fn(
            is_training=True,
            data_dir=flags_obj.data_dir,
            batch_size=flags_obj.batch_size,
            num_epochs=flags_obj.train_epochs,
            parse_record_fn=imagenet_preprocessing.parse_record,
            datasets_num_private_threads=flags_obj.datasets_num_private_threads,
            dtype=dtype,
            drop_remainder=drop_remainder,
            tf_data_experimental_slack=flags_obj.tf_data_experimental_slack,
            training_dataset_cache=flags_obj.training_dataset_cache,
        )

        if flags_obj.cnn_model == 'resnet101':
            model = tf.keras.applications.ResNet101(
                weights=None,
                classes=imagenet_preprocessing.NUM_CLASSES)
        elif flags_obj.cnn_model == 'vgg16':
            model = tf.keras.applications.VGG16(
                weights=None,
                classes=imagenet_preprocessing.NUM_CLASSES)
        elif flags_obj.cnn_model == 'inceptionv3':
            model = tf.keras.applications.InceptionV3(
                weights=None,
                classes=imagenet_preprocessing.NUM_CLASSES)
        elif flags_obj.cnn_model == 'densenet121':
            model = tf.keras.applications.DenseNet121(
                weights=None,
                classes=imagenet_preprocessing.NUM_CLASSES)
        else:
            raise ValueError('Other Model Undeveloped')

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08)

        train_input_iterator = tf.compat.v1.data.make_one_shot_iterator(
            train_input_dataset)
        train_input, train_target = train_input_iterator.get_next()

        steps_per_epoch = (
            imagenet_preprocessing.NUM_IMAGES['train'] // flags_obj.batch_size)
        train_epochs = flags_obj.train_epochs

        if flags_obj.enable_checkpoint_and_export:
            ckpt_full_path = os.path.join(
                flags_obj.model_dir, 'model.ckpt-{epoch:04d}')

        if train_epochs <= 1 and flags_obj.train_steps:
            steps_per_epoch = min(flags_obj.train_steps, steps_per_epoch)
            train_epochs = 1

        num_eval_steps = (
            imagenet_preprocessing.NUM_IMAGES['validation'] //
            flags_obj.batch_size)

        train_output = model(train_input, training=True)
        scc_loss = tf.keras.losses.SparseCategoricalCrossentropy()

        loss = scc_loss(train_target, train_output)
        var_list = variables.trainable_variables() + \
            ops.get_collection(ops.GraphKeys.TRAINABLE_RESOURCE_VARIABLES)
        grad = optimizer.get_gradients(loss, var_list)
        train_op = optimizer.apply_gradients(zip(grad, var_list))

        #####################################################################
        # Create distributed session.
        #   Instead of using the original TensorFlow session for graph execution,
        #   let's use AutoDist's distributed session, in which a computational
        #   graph for distributed training is constructed.
        #
        # [original line]
        # >>> sess = tf.compat.v1.Session()
        #
        sess = autodist.create_distributed_session()
        #####################################################################

        summary = TimeHistory(flags_obj.batch_size, steps_per_epoch)
        for epoch_id in range(train_epochs):
            summary.on_epoch_begin(epoch_id)
            for batch_id in range(steps_per_epoch):
                summary.on_batch_begin(batch_id)
                loss_v, _ = sess.run([loss, train_op])
                summary.on_batch_end(batch_id, loss_v)
            summary.on_epoch_end(epoch_id)
        summary.on_train_end()

    return


def define_imagenet_keras_flags():
    common.define_keras_flags()
    flags_core.set_defaults()
    flags.adopt_module_key_flags(common)


def main(_):
    model_helpers.apply_clean(flags.FLAGS)
    logdir = '/tmp/logs'
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logname = 'imagenet_strategy_{}_model_{}_node_{}_gpu_{}_patch_{}_proxy_{}'.format(
        flags.FLAGS.autodist_strategy,
        flags.FLAGS.cnn_model,
        node_num,
        gpu_num,
        flags.FLAGS.autodist_patch_tf,
        flags.FLAGS.proxy)
    logging.get_absl_handler().use_absl_log_file(logname, logdir)
    with logger.benchmark_context(flags.FLAGS):
        run(flags.FLAGS)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    define_imagenet_keras_flags()
    app.run(main)
