# Copyright 2020 Petuum Inc. All Rights Reserved.
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

"""Predefined simulator with linear model."""
import os
import pickle as pkl

import tensorflow as tf
import numpy as np

from autodist.autosync.simulator.predefined_simulator import PredefinedSimulator
from autodist.proto.synchronizers_pb2 import PSSynchronizer, AllReduceSynchronizer
from autodist.utils import logging


class LinearSimulator(PredefinedSimulator):
    """Simulates strategies for a given graph and resource spec."""

    def __init__(self,
                 graph_item=None,
                 resource_spec=None,
                 batch_size=1,
                 seq_len=1,
                 checkpoint=None):
        super(PredefinedSimulator, self).__init__(graph_item, resource_spec)
        logging.debug('A LinearSimulator is instantiated: batch_size_per_gpu is {}'.format(batch_size))

        self._batch_size_per_gpu = batch_size
        self._seq_len = seq_len

        # For loading weights of the linear model.
        self._checkpoint = checkpoint
        self._weights = None
        if self._checkpoint:
            try:
                self._weights = self.load_checkpoint(checkpoint)
            except ValueError:
                logging.warning('self._checkpoint is invalid')
                self._weights = None

        # TODO(Hao): add the default weights here.
        self._default_weights = ([1] * 12, 1)

    def simulate(self,
                 strategy,
                 graph_item=None,
                 resource_spec=None,
                 checkpoint=None,
                 *args,
                 **kwargs):
        """Return simulated runtime cost given (strategy, graph_item, resource_spec) tuple.

        Args:
            strategy: the strategy to simulate.
            graph_item: the graph_item this strategy is generated on.
            resource_spec: the resource_spec this strategy is on.
            checkpoint: the checkpoint to perform inference (in place of the default weight).

        Returns:
            float: the estimated cost (lower is better).
        """
        if not strategy:
            raise ValueError('strategy is None.')
        if not graph_item:
            if not self._graph_item:
                raise ValueError('No graph item provided.')
            else:
                graph_item = self._graph_item
        if not resource_spec:
            if not self._resource_spec:
                raise ValueError('No resource spec provided.')
            else:
                resource_spec = self._resource_spec

        x = self._extract_feature(strategy, graph_item, resource_spec)

        # The priority of checkpoint lookup priority is:
        # simulate(checkpoint) > self._weight > self._default_weight
        if checkpoint:
            weights = self.load_checkpoint(checkpoint)
        elif self._weights:
            weights = self._weights
        else:
            weights = self._default_weights

        cost = self.inference(np.array(x), weights)
        return cost

    def inference(self, x, weights):
        """

        Args:
            x: features extracts from a (strategy, graph_item, resource_spec).
            weights: trained linear model weight.

        Returns:
            float: ranking score.
        """
        # if not isinstance(inputs, tf.Tensor):
        #     inputs = tf.reshape(tf.convert_to_tensor(inputs), [1, len(inputs)])

        assert len(weights) == 2
        W, b = weights
        cost = np.dot(W, x) + b
        return cost

    def load_checkpoint(self, checkpoint):
        """
        Load a trained weight from a checkpoint.

        Args:
            checkpoint: the file path to a npz, or a list/array of weights.

        Returns:
            list: load weights [W, b].
        """
        logging.info('Loading checkpoint: {}'.format(checkpoint))
        if isinstance(checkpoint, list):
            assert(len(checkpoint) == 2 or len(checkpoint) == 13)
            if len(checkpoint) == 13:
                checkpoint = checkpoint[:11], checkpoint[12]
            return checkpoint
        elif isinstance(checkpoint, str):
            if os.path.isfile(checkpoint):
                weights = np.load(checkpoint)
                return weights['W'], weights['b']
        else:
            raise ValueError('Unable to load the checkpoint: {}'.format(checkpoint))

    def _extract_feature(self,
                         strategy,
                         graph_item,
                         resource_spec):
        """Get the feature vector as input to the linear model."""
        var_name_to_items, resource_item, var_name_to_sync_time = \
            self.extract_prefeature(strategy, graph_item, resource_spec)

        feature_keys = ['transmission', 'network_overhead', 'gpu_kernel_memory_latency']
        ps_server_sync_time = {}
        cc_group_sync_time = {}

        for var_name, var_item in var_name_to_items.items():
            sync_time = var_name_to_sync_time[var_name]

            # Extract per-server and per-group sync time.
            if isinstance(var_item.synchronizer, PSSynchronizer):
                server = var_item.device
                if server not in ps_server_sync_time:
                    ps_server_sync_time[server] = {key: 0.0 for key in feature_keys}
                for key in feature_keys:
                    ps_server_sync_time[server][key] += sync_time[0][key] + sync_time[1][key]
            elif isinstance(var_item.synchronizer, AllReduceSynchronizer):
                group = var_item.group
                if group not in cc_group_sync_time:
                    cc_group_sync_time[group] = {key: 0.0 for key in feature_keys}
                for key in feature_keys:
                    cc_group_sync_time[group][key] += sync_time[key]
            else:
                raise ValueError('Unrecognized type of synchronizer: {}'.format(type(var_item.synchronizer)))

            # Different from predefined modeling, we transform these into feature vectors in this simulator.
            # We care about the sum time of all servers/groups, or the slowest (max) server/group.
            max_ps_server_sync_time = {key: 0.0 for key in feature_keys}
            sum_ps_server_sync_time = {key: 0.0 for key in feature_keys}
            max_cc_group_sync_time = {key: 0.0 for key in feature_keys}
            sum_cc_group_sync_time = {key: 0.0 for key in feature_keys}

            for key in feature_keys:
                max_ps_server_sync_time[key] = \
                    max([sync_time[key] for sync_time in ps_server_sync_time.values()] or [0.0])
                sum_ps_server_sync_time[key] = \
                    sum([sync_time[key] for sync_time in ps_server_sync_time.values()] or [0.0])
                max_cc_group_sync_time[key] = \
                    max([sync_time[key] for sync_time in cc_group_sync_time.values()] or [0.0])
                sum_cc_group_sync_time[key] = \
                    sum([sync_time[key] for sync_time in cc_group_sync_time.values()] or [0.0])

            # concat them to get the feature.
            x = [max_ps_server_sync_time[key] for key in feature_keys] + \
                [sum_ps_server_sync_time[key] for key in feature_keys] + \
                [max_cc_group_sync_time[key] for key in feature_keys] + \
                [sum_cc_group_sync_time[key] for key in feature_keys]
            return x
