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

from collections import OrderedDict

import tensorflow as tf

from autodist.proto.synchronizers_pb2 import PSSynchronizer, AllReduceSynchronizer
from autodist.simulator.base import SimulatorBase
from autodist.simulator.utils import on_same_host, \
    get_dtype_bits
from autodist.utils import logging


class PredefinedSimulator(SimulatorBase):
    """
    Simulator that uses a predefined communication model to estimate the runtime of strategies.

    See this paper TODO(Hao): put the paper link.
    """

    def __init__(self,
                 graph_item=None,
                 resource_spec=None,
                 batch_size=1,
                 seq_len=1,
                 mode='sum'):
        """
        Construct a predefined simulator.

        We need the per-replica batch size and the length of the input sequence to estimate the communication load of
        variables that are sparsely accessed (e.g. embeddings). For dense variables, these two arguments have no
        influence on estimation.
        Note that graph_item and resource_spec are not required to instantiate a simulator object as we allow
        transferring a trained simulator on a graph_item (or resource_spec) to a different graph_item (or different
        resource_spec). This can be done by passing graph_item or resource_spec

        Args:
            graph_item: a GraphItem object, or a path to a serialized GraphItem object.
            resource_spec: a ResourceSpec object, or a path to a resource file.
            batch_size: the per-replica batch size used to train this model, if there are sparse variables.
            seq_len: the average length of input sequences (if there is any).
            mode: use the `sum` or `max` of all variable sync time as the cost.
        """
        super(PredefinedSimulator, self).__init__(graph_item, resource_spec)
        logging.debug('A PredefinedSimualtor is instantiated: batch_size_per_gpu is {}'.format(batch_size))
        self._batch_size_per_gpu = batch_size
        self._seq_len = seq_len
        self._mode = mode

        # Constants for predefined modeling.
        self._network_overhead = 0.0
        self._gpu_kernel_memory_latency = 0.0

    def simulate(self,
                 strategy,
                 graph_item=None,
                 resource_spec=None,
                 *args,
                 **kwargs):
        """
        Return simulated runtime cost given (strategy, graph_item, resource_spec) tuple.

        Args:
            strategy: the strategy to simulate
            graph_item: the graph_item this strategy is generated on.
            resource_spec: the resource_spec this strategy is on.

        Returns:
            float: the estimated runtime (lower is better).
        """
        var_name_to_items, resource_item, var_name_to_sync_time = \
            self.extract_prefeature(strategy, graph_item, resource_spec)

        # Now use the estimated per-variable sync time to calculate the overall sync time.
        ps_server_sync_time = {}
        cc_group_sync_time = {}

        for var_name, var_item in var_name_to_items.items():
            sync_time = var_name_to_sync_time[var_name]

            # we use a simple formula:
            # time = transmission + network_overhead * participating_workers + gpu_memory_latency * max(#gpus)
            if isinstance(var_item.synchronizer, PSSynchronizer):
                server = var_item.device
                if server not in ps_server_sync_time:
                    ps_server_sync_time[server] = 0.0
                send_time = sync_time[0]['transmission'] + \
                            sync_time[0]['network_overhead'] * self._network_overhead + \
                            sync_time[0]['gpu_kernel_memory_latency'] * self._gpu_kernel_memory_latency
                recv_time = sync_time[1]['transmission'] + \
                            sync_time[1]['network_overhead'] * self._network_overhead + \
                            sync_time[1]['gpu_kernel_memory_latency'] * self._gpu_kernel_memory_latency
                # Then accumulate the time for each variable on this PS. Note this is not necessarily accurate as
                # there might exist parallel communication of variables even on one server.
                ps_server_sync_time[server] += send_time
                ps_server_sync_time[server] += recv_time
            elif isinstance(var_item.synchronizer, AllReduceSynchronizer):
                group = var_item.group
                if group not in cc_group_sync_time:
                    # Each group of variables are fused as one message to pass, so we accumulate the
                    # overhead and latency for only ONCE.
                    cc_group_sync_time[group] += sync_time['network_overhead'] * self._network_overhead + \
                        sync_time['gpu_kernel_memory_latency'] * self._gpu_kernel_memory_latency
                cc_group_sync_time[group] += sync_time['transmission']
            else:
                raise ValueError('Unrecognized type of synchronizer: {}'.format(type(var_item.synchronizer)))

        sync_time = [v for v in ps_server_sync_time.values()] + [v for v in cc_group_sync_time.values()]
        if self._mode == 'max':
            # In `max` mode, we assume all PS and collective groups communicate in parallel, and the PS/group that
            # takes the longest time to sync would bound the overall per-iter time.
            per_iter_time = max(sync_time)
        elif self._mode == 'sum':
            # In `sum` mode, we assume all PS and collective groups synchronize sequentially, and the overall per-iter
            # time is the summation of the sync time of all serviers and collective groups.
            # !!Note: both modes have over-simplified assumptions than a real system.
            per_iter_time = sum(sync_time)
        else:
            raise ValueError('Unrecognized simulation mode: {}'.format(self._mode))
        return per_iter_time

    def extract_prefeature(self,
                           strategy,
                           graph_item=None,
                           resource_spec=None):
        """
        Extract impacting factors of the communication time for each variable.

        Args:
            strategy: the strategy to simulate.
            graph_item: the graph_item this strategy is generated for.
            resource_spec: the resource_spec this strategy is on.

        Returns:
            Dict: A dict of variable name (str) to impacting factors (dict).
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
        # TODO(Hao): need to make sure the (strategy, graph_item, resource_spec) match each other.
        # construct the meta objects
        name_to_items, resource_item = self.preprocess(strategy, graph_item, resource_spec)

        # Now estimate the per-variable sync time
        var_sync_time = OrderedDict()
        for var_name, var_item in name_to_items.items():
            if isinstance(var_item.synchronizer, PSSynchronizer):
                var_sync_time[var_name] = self.var_ps_time(var_item, resource_item)
            elif isinstance(var_item.synchronizer, AllReduceSynchronizer):
                var_sync_time[var_name] = self.var_ar_time(var_item, resource_item)
            else:
                raise ValueError('{}'.format(type(var_item.synchronizer)))
        return var_sync_time

    def var_ps_time(self,
                    var_item,
                    resource_item):
        """
        Estimate the synchronization time of a variable that uses PS synchronizer.

        Args:
            var_item: the variable meta information.
            resource_item: the resource meta information.

        Returns:
            tuple(Dict): a dict of potential impacting factors for send and recv time, respectively.
        """
        bits_to_transfer = var_item.bits_to_transfer(self._batch_size_per_gpu, self._seq_len)
        placement = var_item.device
        p2p_bandwidth = resource_item.p2p_bandwidth
        max_num_local_gpu_replica = resource_item.max_num_local_gpu_replica
        num_local_replica_on_each_worker = [resource_item.num_local_gpu_replica_on(host)
                                            for host in resource_item.cpu_replicas]
        if var_item.is_sparse:
            send_time = self._estimate_ps_time(bits_to_transfer,
                                               placement,
                                               p2p_bandwidth,
                                               max_num_local_gpu_replica,
                                               resource_item.cpu_replicas,
                                               num_local_replica_on_each_worker)
            recv_time = self._estimate_ps_time(bits_to_transfer,
                                               placement,
                                               p2p_bandwidth,
                                               max_num_local_gpu_replica,
                                               resource_item.gpu_replicas,
                                               [1.0] * len(resource_item.gpu_replicas))
        else:
            # In AutoDist, the gradients are always locally accumulated then SENT to parameter server.
            send_time = self._estimate_ps_time(bits_to_transfer,
                                               placement,
                                               p2p_bandwidth,
                                               max_num_local_gpu_replica,
                                               resource_item.cpu_replicas,
                                               [1.0] * len(resource_item.cpu_replicas))
            # The communication overhead of receiving parameters from PS depends on `local_replication`.
            if var_item.local_replication:
                recv_time = self._estimate_ps_time(bits_to_transfer,
                                                   placement,
                                                   p2p_bandwidth,
                                                   max_num_local_gpu_replica,
                                                   resource_item.cpu_replicas,
                                                   [1.0] * len(resource_item.cpu_replicas))
            else:
                recv_time = self._estimate_ps_time(bits_to_transfer,
                                                   placement,
                                                   p2p_bandwidth,
                                                   max_num_local_gpu_replica,
                                                   resource_item.gpu_replicas,
                                                   [1.0] * len(resource_item.gpu_replicas))
        return send_time, recv_time

    @staticmethod
    def _estimate_ps_time(bits_to_transfer,
                          placement,
                          p2p_bandwidth,
                          max_num_local_gpu_replica,
                          virtual_worker_list,
                          virtual_num_local_replica):
        """
        Estimate the send or receive time of a ps and return multiple impacting factors.

        Args:
            bits_to_transfer: the variable whose communication time will be estimated.
            placement: the placement of the variable.
            p2p_bandwidth: point-to-point bandwidth between divices of the cluster.
            max_num_local_gpu_replica: the maximum number of on a single node across the cluster.
            virtual_worker_list: A list of virtual workers (could be actual gpu workers, or virtual cpu worker).
            virtual_num_local_replica: A list of integers indicating the number of local replica on each virtual worker.

        Returns:
            Dict: a dictionary of impacting factors.
        """
        transmission_time = 0.0

        # To estimate network transmission time for the given variable var_item on PS, we simply sum up the time of
        # transmitting (or say, synchronizing) this variable across all workers.
        # The time is separately estimated as send_time and recv_time by calling this function twice with different
        # values of arguments.
        # TODO(Hao): didn't consider any parallelization between variables or partitions.
        for k, worker in enumerate(virtual_worker_list):
            if not on_same_host(placement, worker):
                bits_on_this_worker = bits_to_transfer * virtual_num_local_replica[k]
                bandwidth = min(p2p_bandwidth[placement][worker], p2p_bandwidth[worker][placement])
                transmission_time += bits_on_this_worker / bandwidth
        factors = {
            'transmission': transmission_time,
            'network_overhead': len(virtual_worker_list),
            'gpu_kernel_memory_latency': max_num_local_gpu_replica,  # TODO(Hao): Is this correct?
            'constant': 1.0
        }
        return factors

    def var_ar_time(self,
                    var_item,
                    resource_item,
                    powersgd_rank=10):
        """
        Estimate the synchronization time of a variable that uses collective synchronizer.

        Due to limitation, we only consider dense variables for now.
        Args:
            var_item: the variable meta information.
            resource_item: the resource meta information.

        Returns:
            Dict: a dictionary of impacting factors.
        """
        # Address cases for different types of compressors
        if var_item.compressor not in ['PowerSGDCopmressor', 'HorovodCompressorEF', 'HorovodCompressor',
                                       'NoneCompressor', 0, 1, 2, 3]:
            raise ValueError('Compressor type not recognized: {}'.format(var_item.compressor))

        size_to_transfer = var_item.size_to_transfer(batch_size_per_gpu=self._batch_size_per_gpu,
                                                     seq_len=self._seq_len)
        dtype = var_item.dtype

        if var_item.compressor in ['PowerSGDCopmressor', 3, "HorovodCompressorEF", "HorovodCompressor", 1, 2]:
            # These compressors always use float32 to communicate.
            dtype = tf.float32
        if var_item.compressor in ["PowerSGDCompressor", 3]:
            # For PowerSGDCompessor, we hard-code the rank as 10. It will always use float32 to communicate.
            if len(var_item.shape) > 1:
                n = var_item.shape[0]
                m = 1
                for d in var_item.shape[1:]:
                    m *= d
                size_to_transfer = (m + n) * powersgd_rank

        # We assume ring allreduce, and multiple rings will be constructed and executed serialliy to synchronize grads.
        # In one ring, each worker exchanges grads with its next worker in parallel. Hence, the time a single ring
        # completes is bounded by the slowest pair of workers; the total time spent for all workers to synchronize
        # grads are bounded by the time all rings finish on the slowest pair of workers.
        transmission_time = size_to_transfer * get_dtype_bits(dtype) / resource_item.min_bandwidth
        factors = {
            'transmission': transmission_time,
            'network_overhead': 1,  # TODO(Hao): is this correct?
            'gpu_kernel_memory_latency': resource_item.max_num_local_gpu_replica,
            'constant': 1.0
        }
        return factors
