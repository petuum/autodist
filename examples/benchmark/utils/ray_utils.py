# Copyright 2021 Petuum, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import sys
import os
import time
import ray
import numpy as np
import tensorflow as tf

from autodist.strategy import PS, PSLoadBalancing, PartitionedPS, AllReduce, Parallax
from autodist.ray import TFTrainer, TFRunner

def run_ray_job(strategy,
                model_fn,
                loss_fn,
                model_dir,
                train_input_fn,
                steps_per_epoch,
                steps_per_loop,
                epochs,
                sub_model_export_name,
                custom_callbacks):

    def _get_input_iterator(input_fn, strategy):
        """Returns distributed dataset iterator."""
        # When training with TPU pods, datasets needs to be cloned across
        # workers. Since Dataset instance cannot be cloned in eager mode, we instead
        # pass callable that returns a dataset.
        if not callable(input_fn):
            raise ValueError(
                '`input_fn` should be a closure that returns a dataset.')
        if not isinstance(strategy, tf.distribute.Strategy):
            iterator = tf.compat.v1.data.make_one_shot_iterator(input_fn())
        else:
            iterator = iter(
                strategy.experimental_distribute_datasets_from_function(input_fn))
        return iterator

    def _replicated_step(model, core_model, inputs):
        """Replicated training step."""
        optimizer = model.optimizer
        use_float16 = isinstance(
            optimizer, tf.keras.mixed_precision.experimental.LossScaleOptimizer)

        inputs, labels = inputs
        model_outputs = model(inputs, training=True)
        loss = loss_fn(labels, model_outputs)
        if use_float16:
            scaled_loss = optimizer.get_scaled_loss(loss)

        training_vars = model.trainable_variables
        if use_float16:
            scaled_grads = tf.gradients(scaled_loss, training_vars)
            grads = optimizer.get_unscaled_gradients(scaled_grads)
        else:
            grads = tf.gradients(loss, training_vars)
        train_op = optimizer.apply_gradients(zip(grads, training_vars))
        return train_op, loss

    def input_fn():
        return _get_input_iterator(train_input_fn, strategy)

    trainer = TFTrainer(strategy, _replicated_step, model_fn, input_fn)

    for epoch in range(2):
        per_replica = trainer.train()
        for host, output in per_replica.items():
            _, l = output
            print(f"node:{host}\tloss: {l}")

    trainer.shutdown()


