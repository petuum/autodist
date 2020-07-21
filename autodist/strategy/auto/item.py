# Copyright 2020 Petuum. All Rights Reserved.
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

"""Helper classes and functions for automatic strategy generation."""

from enum import Enum

from tensorflow.python.framework import ops, device_spec

from autodist.kernel.common.utils import get_op_name, get_consumers
from autodist.kernel.device.resolver import DeviceResolver
from autodist.graph_item import cached_property
from autodist.strategy.base import byte_size_load_fn
from autodist.utils import logging
from autodist.cluster import SSHCluster
from autodist.simulator.utils import GPU_TO_CPU_BANDWIDTH, GIGABITS


class VarType(Enum):
    SPARSE = 0
    DENSE = 1


class VariableItem:
    """Helper class to include meta information about a variable."""
    def __init__(self,
                 var,
                 graph_item,
                 node_config=None):
        self.var = var
        self.graph_item = graph_item
        self._var_op_name = get_op_name(var.name)
        self._grad = graph_item.var_op_name_to_grad_info[self._var_op_name][0]

        self._config = None
        if node_config:
            self.update_config(node_config)
        else:
            logging.warning('Item with name {} has empty config.'.format(self.name))

    def update_config(self, config):
        """
        Update the nodeconfig of this variable.

        Args:
            config:
        """
        assert not config
        self._node_config = config

    @property
    def var_type(self):
        """
        Return the type of the variable (VarType.SPARSE or VarType.DENSE).

        Returns:
            VarType
        """
        return VarType.DENSE if isinstance(self._grad, ops.Tensor) else VarType.SPARSE

    @property
    def name(self):
        """
        Return the name of the variable.

        Returns:
            String
        """
        return self.var.name

    @property
    def is_sparse(self):
        """
        Return whether the variable is sparse.

        Returns:
            Bool
        """
        return True if self.var_type == VarType.SPARSE else False

    @property
    def is_embedding(self):
        """
        Return whether the variable corresponds to an embedding.

        Returns:
            Bool
        """
        # TODO (Hao): better way to determine is_embedding?
        for op in get_consumers(self.var.op):
            if op.type == "ResourceGather":
                return True
        return False

    @property
    def shape(self):
        """
        Return the shape of the variable, or None if it does not emit a tensor (e.g. scalar).

        Returns:
            List(int)
        """
        return self.original_shape

    @property
    def original_shape(self):
        if self.var.initial_value.shape.ndims:
            return self.var.initial_value.shape.as_list()
        else:
            return None

    @property
    def size(self):
        size = 1
        if self.shape:
            for s in self.shape:
                size *= s
        return size

    @property
    def original_size(self):
        size = 1
        if self.original_shape:
            for s in self.original_shape:
                size *= s
        return size

    @property
    def size_to_transfer(self, batch_size_per_gpu=1, seq_len=1):
        if not self.is_sparse:
            return self.size
        else:
            if not self.shape: # scalar
                return 1

            emb_size = 1
            if len(self.shape) > 1:
                # infer the embedding size from original shape
                for i in range(1, len(self.original_shape)):
                    emb_size *= self.original_shape[i]

            sparse_data_size = batch_size_per_gpu * seq_len * emb_size

            # estimate the embedding of this partition simply using a proportional formula
            return sparse_data_size * self.size / self.original_size

    @property
    def partitionable_axes(self):
        """
        Return the list of available axes that are legitimate to partition along.

        Returns:
            List(int)
        """
        valid_axes = []

        # scalar
        if not self.shape:
            return valid_axes

        # Sparse variable can only be partition along the 0th axis in current implementation.
        if self.is_sparse or self.is_embedding:
            valid_axes = [0]
            return valid_axes
        for idx, dim in enumerate(self.shape):
            if dim > 1:
                valid_axes.append(idx)
        return valid_axes

    @property
    def byte_size(self):
        """
        Return the byte size of the variable.

        Returns:
            float
        """
        return float(byte_size_load_fn(self.var))

    @property
    def dtype(self):
        """
        Return the dtype of the variable.

        Returns:
            dtype
        """
        return self.var.dtype

    @property
    def synchronizer(self):
        """
        Return the synchronizer protobuf in the config of this variable.

        Returns:
            NodeConfig
        """
        if not self._node_config:
            raise ValueError('Node config is unset.')
        if self._node_config.partitioner:
            logging.warning('This variable will be partitioned')
            return None
        return getattr(self._node_config, self._node_config.WhichOneOf('synchronizer'))

    @property
    def compressor(self):
        """
        Return the compressor in the node config of this variable.

        Returns:
            Compressor type.
        """
        if not self._node_config:
            raise ValueError('Node config is unset.')
        if self._node_config.partitioner:
            logging.warning('This variable will be partitioned')
            return None
        return getattr(self.synchronizer, 'compressor', None)

    @property
    def reduction_destination(self):
        """
        Return the reduction_destination in the node config of this variable.

        Returns:
            Reduction destinaiton.
        """
        if not self._node_config:
            raise ValueError('Node config is unset.')
        if self._node_config.partitioner:
            logging.warning('This variable will be partitioned')
            return None
        return getattr(self.synchronizer, 'reduction_destination', None)

    def device(self, resolver):
        device_str = self.reduction_destination if self.reduction_destination else self.var.device
        if device_str:
            device_str =  resolver.resolve_to_device_str(device_str)
        return device_str

class PartItem(VariableItem):
    """Helper class to include meta information about a variable partition."""
    def __init__(self,
                 var,
                 graph_item,
                 part_idx,
                 pc,
                 part_config=None):
        super(PartItem, self).__init__(var, graph_item, part_config)

        self.part_idx = part_idx
        self.pc = pc

    @property
    def name(self):
        """
        Return the name of this partition.

        Returns:
            String
        """
        name = '{}/part_{}:0'.format(get_op_name(self.var.name), self.part_idx)
        return name

    @property
    def partition_str(self):
        return self.pc.partition_str

    @property
    def shape(self):
        """
        Return the shape of this partition.

        Returns:
            List(int)

        """
        shape = self.original_shape
        if shape:
            dim_size = shape[self.pc.axis] // self.pc.num_shards
            extras = shape[self.pc.axis] % self.pc.num_shards
            if self.part_idx < extras:
                dim_size += 1
            shape[self.pc.axis] = dim_size
        return shape

    @property
    def partitionable_axes(self):
        """
        Return the list of available axes that are legitimate to partition along.

        Returns:
            None: because this is a partition (not allowed to be partitioned further).
        """
        return []

    @property
    def byte_size(self):
        """
        Return the byte size of this partition.

        Returns:
            float
        """
        return float(byte_size_load_fn(self.var)) \
            * float(self.shape[self.pc.axis]) / float(self.original_shape[self.pc.axis])

    @property
    def synchronizer(self):
        """

        Returns:

        """
        if not self._node_config:
            raise ValueError('Node config is unset.')
        if not self._node_config.partitioner:
            raise ValueError('Partitioner field is empty for a variable partition.')
        return getattr(self._node_config, self._node_config.WhichOneOf('synchronizer'))

    @property
    def compressor(self):
        """
        Return the compressor in the node config of this variable partition.

        Returns:
            Compressor.
        """
        if not self._node_config:
            raise ValueError('Node config is unset.')
        if not self._node_config.partitioner:
            raise ValueError('Partitioner field is empty for a variable partition.')
        return getattr(self.synchronizer, 'compressor', None)

    @property
    def reduction_destination(self):
        """
        Return the reduction_destination in the node config of this variable partition.

        Returns:
            Reduction destination.
        """
        if not self._node_config:
            raise ValueError('Node config is unset.')
        if not self._node_config.partitioner:
            logging.warning('Partitioner field is empty for a variable partition.')
            return None
        return getattr(self.synchronizer, 'reduction_destination', None)


class ResourceItem:
    """ResourceItem.

    Helper class that includes meta information about a resource spec. All addresses are resolved (in TF format).

    TODO(zhisbug): merge ResourceItem class with ResourceSpec.
    """

    def __init__(self, resource_spec):
        self._resource_spec = resource_spec
        self._cluster = SSHCluster(resource_spec)
        self._device_resolver = DeviceResolver(self._cluster)

    @property
    def replicas(self):
        """Return the list of replicas in the format of TF device string, e.g. job:worker/task:0/device:gpu:0."""
        device_strs = [k for k, _ in self._resource_spec.devices]
        return self._device_resolver.resolve_to_device_str(device_strs)

    @property
    def gpu_replicas(self):
        """
        Return the list of GPU replicas in the format of TF device string, e.g. job:worker/task:0/device:gpu:0.

        Returns:
            List(string)
        """
        # device_str is autodist device string, e.g. 192.168.0.1:CPU:0
        device_strs = [k for k, _ in self._resource_spec.gpu_devices]
        return self._device_resolver.resolve_to_device_str(device_strs)

    @property
    def cpu_replicas(self):
        """
        Return the list of CPU replicas in the format of TF device string, e.g. job:worker/task:0/device:cpu:0.

        Returns:
            List(string)
        """
        device_strs = [k for k, _ in self._resource_spec.cpu_devices]
        return self._device_resolver.resolve_to_device_str(device_strs)

    @property
    def total_num_gpu_replica(self):
        return len(self.gpu_replicas)

    def num_local_gpu_replica(self, host):
        """
        Return the number of gpu replica on a TF host address, e.g. '/job:worker/task:0/device:CPU:0'.

        Args:
            host: TF host address,e .g. '/job:worker/task:0/device:CPU:0'

        Returns:
            int
        """
        gpu_device_specs = {device_spec.DeviceSpecV2.from_string(d) for d in self.gpu_replicas}
        num = 0
        host_device_spec = device_spec.DeviceSpecV2.from_string(host)
        for d in gpu_device_specs:
            if self._cluster.get_address_from_task(d.job, d.task) \
                 == self._cluster.get_address_from_task(host_device_spec.job, host_device_spec.task):
                num += 1
        return num

    @property
    def max_num_local_gpu_replica(self):
        """Return the max number of local gpu replicas on the cluster."""
        return max([self.num_local_gpu_replica(host) for host in self.cpu_replicas])

    @cached_property
    def p2p_bandwidth(self):
        """Calculates P2P network bandwidth between nodes in the cluster.

        Note that this is NOT a sysmetric
        """
        bw = {} # key: (device1, device2)
        devices = [device for device, _ in self._resource_spec.devices]
        resolved_devices = self.replicas

        for i in range(len(self.replicas)):
            ip_i = devices[i].split(':')[0]
            d_i = resolved_devices[i]
            if d_i not in bw:
                bw[d_i] = {}
            for j in range(i, len(self.replicas)):
                ip_j = devices[j].split(':')[0]
                d_j = resolved_devices[j]
                if d_j not in bw:
                    bw[d_j] = {}
                if ip_i != ip_j:
                    bw[d_i][d_j] = GIGABITS * self._resource_spec[ip_i].bandwidth[ip_i]
                    bw[d_j][d_i] = GIGABITS * self._resource_spec[ip_j].bandwidth[ip_j]
                else:
                    bw[d_i][d_j] = GIGABITS * GPU_TO_CPU_BANDWIDTH
                    bw[d_j][d_i] = GIGABITS * GPU_TO_CPU_BANDWIDTH
        return bw

    @cached_property
    def min_bandwidth(self):
        """Return the minimum bandwidth (bottleneck) of all p2p connections on this cluster."""
        return min([min(v.values()) for k, v in self.p2p_bandwidth])
