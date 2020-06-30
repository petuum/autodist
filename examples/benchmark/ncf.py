# Copyright 2020 Petuum. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import yaml
import os
import sys
import tensorflow as tf
# pylint: disable=g-bad-import-order
from absl import app
from absl import flags
from absl import logging
from os.path import expanduser
from tensorflow_addons.optimizers import LazyAdam

# pylint: enable=g-bad-import-order
from utils.recommendation import constants as rconst
from utils.recommendation import movielens
from utils.recommendation import ncf_common
from utils.recommendation import ncf_input_pipeline
from utils.recommendation import neumf_model

from utils.flags import core as flags_core
from utils.logs import logger
from utils.logs import mlperf_helper
from utils.misc import keras_utils
from utils.misc import model_helpers

#########################################################################
# Import AutoDist and Strategy
from autodist import AutoDist
from autodist.strategy.ps_strategy import PS
from autodist.strategy.ps_lb_strategy import PSLoadBalancing
from autodist.strategy.partitioned_ps_strategy import PartitionedPS
from autodist.strategy.all_reduce_strategy import AllReduce
from autodist.strategy.parallax_strategy import Parallax
#########################################################################

FLAGS = flags.FLAGS


class IncrementEpochCallback(tf.keras.callbacks.Callback):
    """A callback to increase the requested epoch for the data producer.

    The reason why we need this is because we can only buffer a limited amount of
    data. So we keep a moving window to represent the buffer. This is to move the
    one of the window's boundaries for each epoch.
    """

    def __init__(self, producer):
        self._producer = producer

    def on_epoch_begin(self, epoch, logs=None):
        self._producer.increment_request_epoch()


class CustomEarlyStopping(tf.keras.callbacks.Callback):
    """Stop training has reached a desired hit rate."""

    def __init__(self, monitor, desired_value):
        super(CustomEarlyStopping, self).__init__()

        self.monitor = monitor
        self.desired = desired_value
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current and current >= self.desired:
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            logging.warning(
                "Early stopping conditioned on metric `%s` "
                "which is not available. Available metrics are: %s",
                self.monitor,
                ",".join(
                    list(
                        logs.keys())))
        return monitor_value


def _get_keras_model(params):
    """Constructs and returns the model."""
    batch_size = params["batch_size"]
    user_input = tf.keras.layers.Input(
        shape=(1,), name=movielens.USER_COLUMN, dtype=tf.int32)
    item_input = tf.keras.layers.Input(
        shape=(1,), name=movielens.ITEM_COLUMN, dtype=tf.int32)
    valid_pt_mask_input = tf.keras.layers.Input(
        shape=(1,), name=rconst.VALID_POINT_MASK, dtype=tf.bool)
    dup_mask_input = tf.keras.layers.Input(
        shape=(1,), name=rconst.DUPLICATE_MASK, dtype=tf.int32)
    label_input = tf.keras.layers.Input(
        shape=(1,), name=rconst.TRAIN_LABEL_KEY, dtype=tf.bool)
    base_model = neumf_model.construct_model(user_input, item_input, params)
    logits = base_model.output
    zeros = tf.keras.layers.Lambda(lambda x: x * 0)(logits)
    softmax_logits = tf.keras.layers.concatenate([zeros, logits], axis=-1)
    keras_model = tf.keras.Model(
        inputs={
            movielens.USER_COLUMN: user_input,
            movielens.ITEM_COLUMN: item_input,
            rconst.VALID_POINT_MASK: valid_pt_mask_input,
            rconst.DUPLICATE_MASK: dup_mask_input,
            rconst.TRAIN_LABEL_KEY: label_input},
        outputs=softmax_logits)
    keras_model.summary()
    return keras_model


def run_ncf(FLAGS):
    """Run NCF training and eval with Keras."""

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

    if FLAGS.autodist_patch_tf:
        os.environ['AUTODIST_PATCH_TF'] = '1'
    else:
        os.environ['AUTODIST_PATCH_TF'] = '0'

    if FLAGS.proxy:
        local_proxy_variable = True
    else:
        local_proxy_variable = False

    if FLAGS.autodist_strategy == 'PS':
        autodist = AutoDist(
            resource_spec_file, PS(
                local_proxy_variable=local_proxy_variable))
    elif FLAGS.autodist_strategy == 'PSLoadBalancing':
        autodist = AutoDist(resource_spec_file, PSLoadBalancing(
            local_proxy_variable=local_proxy_variable))
    elif FLAGS.autodist_strategy == 'PartitionedPS':
        autodist = AutoDist(resource_spec_file, PartitionedPS(
            local_proxy_variable=local_proxy_variable))
    elif FLAGS.autodist_strategy == 'AllReduce':
        autodist = AutoDist(resource_spec_file, AllReduce(chunk_size=256))
    elif FLAGS.autodist_strategy == 'Parallax':
        autodist = AutoDist(
            resource_spec_file,
            Parallax(
                chunk_size=256,
                local_proxy_variable=local_proxy_variable))
    else:
        raise ValueError(
            'the strategy can be only from PS, PSLoadBalancing, PartitionedPS, AllReduce, Parallax')
    #########################################################################
    if FLAGS.seed is not None:
        print("Setting tf seed")
        tf.random.set_seed(FLAGS.seed)

    model_helpers.apply_clean(FLAGS)

    if FLAGS.dtype == "fp16" and FLAGS.fp16_implementation == "keras":
        policy = tf.keras.mixed_precision.experimental.Policy(
            "mixed_float16", loss_scale=flags_core.get_loss_scale(
                FLAGS, default_for_fp16="dynamic"))
        tf.keras.mixed_precision.experimental.set_policy(policy)

    params = ncf_common.parse_flags(FLAGS)
    params["distribute_strategy"] = None

    batch_size = params["batch_size"]
    time_callback = keras_utils.TimeHistory(batch_size, FLAGS.log_steps)
    callbacks = [time_callback]

    producer, input_meta_data = None, None
    generate_input_online = params["train_dataset_path"] is None

    if generate_input_online:
        num_users, num_items, _, _, producer = ncf_common.get_inputs(params)
        producer.start()
        per_epoch_callback = IncrementEpochCallback(producer)
        callbacks.append(per_epoch_callback)
    else:
        assert params["eval_dataset_path"] and params["input_meta_data_path"]
        with tf.io.gfile.GFile(params["input_meta_data_path"], "rb") as reader:
            input_meta_data = json.loads(reader.read().decode("utf-8"))
            num_users = input_meta_data["num_users"]
            num_items = input_meta_data["num_items"]

    params["num_users"], params["num_items"] = num_users, num_items

    if FLAGS.early_stopping:
        early_stopping_callback = CustomEarlyStopping(
            "val_HR_METRIC", desired_value=FLAGS.hr_threshold)
        callbacks.append(early_stopping_callback)

    with tf.Graph().as_default(), autodist.scope():
        (train_input_dataset, eval_input_dataset, num_train_steps, num_eval_steps) = (
            ncf_input_pipeline.create_ncf_input_data(params, producer, input_meta_data, None))
        steps_per_epoch = None if generate_input_online else num_train_steps
        keras_model = _get_keras_model(params)
        if FLAGS.optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=params["learning_rate"],
                beta_1=params["beta1"],
                beta_2=params["beta2"],
                epsilon=params["epsilon"])
        elif FLAGS.optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=params["learning_rate"])
        elif FLAGS.optimizer == 'lazyadam':
            optimizer = LazyAdam(
                learning_rate=params["learning_rate"],
                beta_1=params["beta1"],
                beta_2=params["beta2"],
                epsilon=params["epsilon"])
        else:
            raise ValueError('Do not support other optimizers...')
        if FLAGS.fp16_implementation == "graph_rewrite":
            optimizer = tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(
                optimizer, loss_scale=flags_core.get_loss_scale(FLAGS, default_for_fp16="dynamic"))
        elif FLAGS.dtype == "fp16" and params["keras_use_ctl"]:
            optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
                optimizer, tf.keras.mixed_precision.experimental.global_policy().loss_scale)

        return run_ncf_custom_training(
            params,
            autodist,
            keras_model,
            optimizer,
            callbacks,
            train_input_dataset,
            eval_input_dataset,
            num_train_steps,
            num_eval_steps,
            generate_input_online=generate_input_online,
            return_simulation=FLAGS.simulation_strategy_id is not None)


def run_ncf_custom_training(params,
                            autodist,
                            keras_model,
                            optimizer,
                            callbacks,
                            train_input_dataset,
                            eval_input_dataset,
                            num_train_steps,
                            num_eval_steps,
                            generate_input_online=True,
                            return_simulation=False):
    """Runs custom training loop.

    Args:
        params: Dictionary containing training parameters.
        strategy: Distribution strategy to be used for distributed training.
        keras_model: Model used for training.
        optimizer: Optimizer used for training.
        callbacks: Callbacks to be invoked between batches/epochs.
        train_input_dataset: tf.data.Dataset used for training.
        eval_input_dataset: tf.data.Dataset used for evaluation.
        num_train_steps: Total number of steps to run for training.
        num_eval_steps: Total number of steps to run for evaluation.
        generate_input_online: Whether input data was generated by data producer.
            When data is generated by data producer, then train dataset must be
            re-initialized after every epoch.

    Returns:
        A tuple of train loss and a list of training and evaluation results.
    """
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction="sum", from_logits=True)
    train_input_iterator = tf.compat.v1.data.make_one_shot_iterator(
        train_input_dataset).get_next()

    def step_fn(features):
        softmax_logits = keras_model(features)
        softmax_logits = tf.cast(softmax_logits, "float32")
        labels = features[rconst.TRAIN_LABEL_KEY]
        loss = loss_object(
            labels,
            softmax_logits,
            sample_weight=features[rconst.VALID_POINT_MASK])
        loss *= (1.0 / params["batch_size"])
        if FLAGS.dtype == "fp16":
            loss = optimizer.get_scaled_loss(loss)

        grads = tf.gradients(loss, keras_model.trainable_variables)
        if FLAGS.dtype == "fp16":
            grads = optimizer.get_unscaled_gradients(grads)
        grads_and_vars = list(zip(grads, keras_model.trainable_variables))
        if FLAGS.dense_gradient:
            grads_and_vars = neumf_model.sparse_to_dense_grads(grads_and_vars)
        train_op = optimizer.apply_gradients(grads_and_vars)
        return loss, train_op, keras_model.trainable_variables, grads, optimizer

    for callback in callbacks:
        callback.on_train_begin()

    if FLAGS.ml_perf:
        eval_summary_writer, train_summary_writer = None, None
    else:
        summary_dir = os.path.join(FLAGS.model_dir, "summaries")
        eval_summary_writer = tf.summary.create_file_writer(
            os.path.join(summary_dir, "eval"))
        train_summary_writer = tf.summary.create_file_writer(
            os.path.join(summary_dir, "train"))

    loss_op, train_op, vars, grads, optimizer = step_fn(train_input_iterator)
    #####################################################################
    # Create distributed session.
    #   Instead of using the original TensorFlow session for graph execution,
    #   let's use AutoDist's distributed session, in which a computational
    #   graph for distributed training is constructed.
    #
    # [original line]
    # >>> sess = tf.compat.v1.Session()
    sess = autodist.create_distributed_session()
    #####################################################################
    for epoch in range(FLAGS.num_epochs):
        for cb in callbacks:
            cb.on_epoch_begin(epoch)

        train_loss = 0
        for step in range(FLAGS.trial_steps):
            current_step = step + epoch * num_train_steps
            for c in callbacks:
                c.on_batch_begin(current_step)

            loss_val, _ = sess.run([loss_op, train_op])
            train_loss += loss_val

            if train_summary_writer and step % 1000 == 0:
                with train_summary_writer.as_default():
                    tf.summary.scalar(
                        "training_loss", train_loss / (step + 1), step=current_step)

            for c in callbacks:
                c.on_batch_end(current_step)

        train_loss /= num_train_steps
        logging.info(
            "Done training epoch %s, epoch loss=%s.",
            epoch + 1,
            train_loss)

    for c in callbacks:
        c.on_train_end()

    return train_loss


def build_stats(loss, eval_result, time_callback):
    """Normalizes and returns dictionary of stats.

    Args:
        loss: The final loss at training time.
        eval_result: Output of the eval step. Assumes first value is eval_loss and
            second value is accuracy_top_1.
        time_callback: Time tracking callback likely used during keras.fit.

    Returns:
        Dictionary of normalized results.
    """
    stats = {}
    if loss:
        stats["loss"] = loss

    if eval_result:
        stats["eval_loss"] = eval_result[0]
        stats["eval_hit_rate"] = eval_result[1]

    if time_callback:
        timestamp_log = time_callback.timestamp_log
        stats["step_timestamp_log"] = timestamp_log
        stats["train_finish_time"] = time_callback.train_finish_time
        if len(timestamp_log) > 1:
            stats["avg_exp_per_second"] = (
                time_callback.batch_size * time_callback.log_steps *
                (len(time_callback.timestamp_log) - 1) /
                (timestamp_log[-1].timestamp - timestamp_log[0].timestamp))

    return stats


def define_trial_run_flags():
    flags.DEFINE_string(
        name='resource',
        default='',
        help='resource specification')
    flags.DEFINE_integer(
        name='trial_steps',
        default=100,
        help='number of steps for trial')
    flags.DEFINE_enum(
        name="optimizer",
        default="adam",
        enum_values=[
            "adam",
            "lazyadam",
            "sgd"],
        case_sensitive=False,
        help=flags_core.help_wrap("optimizer to use."))
    flags.DEFINE_bool(
        name='dense_gradient',
        default='True',
        help='whether to use dense gradient')
    flags.DEFINE_string(
        name='simulation_strategy_id',
        default=None,
        help='strategy id to simulate')
    flags.DEFINE_string(
        name='simulation_folder',
        default=None,
        help='folder to store simulation result')
    flags.DEFINE_string(
        name='autodist_strategy',
        default='PS',
        help='the autodist strategy')
    flags.DEFINE_boolean(
        name='autodist_patch_tf',
        default=True,
        help='AUTODIST_PATCH_TF')
    flags.DEFINE_boolean(name='proxy', default=True, help='proxy')
    flags.DEFINE_string(
        name='default_data_dir',
        default='~/movielens',
        help='the default data directory')
    flags.DEFINE_integer(
        name='num_epochs',
        default=3,
        help='number of training epochs')


def main(_):
    logdir = '/tmp/logs'
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logname = 'ncf_strategy_{}_opt_{}_dense_{}'.format(
        FLAGS.autodist_strategy, FLAGS.optimizer, FLAGS.dense_gradient)
    logging.get_absl_handler().use_absl_log_file(logname, logdir)
    with logger.benchmark_context(FLAGS), mlperf_helper.LOGGER(FLAGS.output_ml_perf_compliance_logging):
        mlperf_helper.set_ncf_root(os.path.split(os.path.abspath(__file__))[0])
        FLAGS.keras_use_ctl = True
        FLAGS.run_eagerly = False
        FLAGS.eval_batch_size = 1000
        FLAGS.dataset = 'ml-20mx16x32'
        FLAGS.train_dataset_path = os.path.join(
            FLAGS.default_data_dir,
            FLAGS.dataset,
            'tfrecord/training_cycle_0/*')
        FLAGS.eval_dataset_path = os.path.join(
            FLAGS.default_data_dir, FLAGS.dataset, 'tfrecord/eval_data/*')
        FLAGS.input_meta_data_path = os.path.join(
            FLAGS.default_data_dir, FLAGS.dataset, 'tfrecord/meta')
        run_ncf(FLAGS)


if __name__ == "__main__":
    ncf_common.define_ncf_flags()
    define_trial_run_flags()
    app.run(main)
