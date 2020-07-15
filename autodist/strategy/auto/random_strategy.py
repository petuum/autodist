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

"""AllReduce StrategyBuilder."""
from collections import OrderedDict

from enum import Enum
from tensorflow.python.framework import ops

from arion.kernel.common.utils import get_op_name, get_consumers
from arion.kernel.partitioner import PartitionerConfig
from arion.proto import strategy_pb2, synchronizers_pb2
from arion.search import sample_util
from arion.strategy.base import Strategy, StrategyBuilder
from arion.strategy.base import byte_size_load_fn
from arion.strategy.component.ps_load_balancer import greedy_load_balancer, christy_load_balancer
from arion.strategy.component.ar_group_assigner import chunk_group_assigner, \
    christy_group_assigner, ordered_balanced_group_assigner

class VarType(Enum):
    SPARSE = 0
    DENSE = 1


class VariableHelper:
    def __init__(self, var, graph_item):
        self.var = var
        self.graph_item = graph_item
        self._var_op_name = get_op_name(var.name)
        self._grad = graph_item.var_op_name_to_grad_info[self._var_op_name][0]

    @property
    def var_type(self):
        return VarType.DENSE if isinstance(self._grad, ops.Tensor) else VarType.SPARSE

    @property
    def is_sparse(self):
        return True if self.var_type == VarType.SPARSE else False

    @property
    def is_embedding(self):
        for op in get_consumers(self.var.op):
            if op.type == "ResourceGather":
                return True
            # op = new_graph_item.graph.get_operation_by_name(
            #     ops.prepend_name_scope(op.name, ARION_TO_DELETE_SCOPE)
            # )
        return False

    @property
    def shape(self):
        if self.var.initial_value.shape.ndims:
            return self.var.initial_value.shape.as_list()
        else:
            return None

    @property
    def partitionable_axis(self):
        valid_axis = []
        if not self.shape:
            return valid_axis
        # Sparse variable can only be partition along the 0th axis
        # only sample axis for dense variables
        if self.is_sparse or self.is_embedding:
            valid_axis = [0]
            return valid_axis
        for idx, dim in enumerate(self.shape):
            if dim > 1:
                valid_axis.append(idx)
        return valid_axis

    @property
    def byte_size(self):
        return float(byte_size_load_fn(self.var))

    @property
    def dtype(self):
        return self.var.dtype


class PartHelper:
    def __init__(self, part_idx, var, pc):
        self.var = var
        self.part_idx = part_idx
        self.pc = pc

    @property
    def shape(self):
        shape = self.var.initial_value.shape.as_list()
        dim_size = shape[self.pc.axis] // self.pc.num_shards
        extras = shape[self.pc.axis] % self.pc.num_shards
        if self.part_idx < extras:
            dim_size += 1
        shape[self.pc.axis] = dim_size
        return shape

    @property
    def var_shape(self):
        return self.var.initial_value.shape.as_list()

    @property
    def byte_size(self):
        return float(byte_size_load_fn(self.var)) \
               * float(self.shape[self.pc.axis]) / float(self.var_shape[self.pc.axis])


class RandomStrategy(StrategyBuilder):
    def __init__(self, space, heuristics):
        """

        Args:
            self:
            enable_ps_load_balancer:
            enable_chunk:

        Returns:

        """
        self.space = space
        self.heuristics = heuristics
        self.helpers = {}

    def reset(self):
        self.helpers = {}

    def build(self, graph_item, resource_spec):
        expr = Strategy()

        # number of graph replica is equal to number of GPU devices
        expr.graph_config.replicas.extend([k for k, v in resource_spec.gpu_devices])
        variables = graph_item.trainable_var_op_to_var.values()

        # A fully MCMC process to generate node configs
        node_config = []
        for var in variables:
            var_helper = VariableHelper(var, graph_item)
            self.helpers[var_helper.var.name] = var_helper

            node = strategy_pb2.Strategy.Node()
            node.var_name = var_helper.var.name

            # Step 1: determine whether or not to partition
            # TODO(Hao): other factor not considered -- number of reduction_device_names
            maybe_partition = sample_if_partition(var_helper, resource_spec, self.space, self.heuristics)

            # Step 2.1: if not partition, sample a synchronizer type for it
            if not maybe_partition:  # no partition
                sample_var_synchronizer(node, var_helper, resource_spec, self.space)
            else:  # Step 2.2: if partition
                # Step 2.2.1: sample a partitioner config
                pc = sample_partition_config(var_helper, resource_spec, self.space, self.heuristics)
                node.partitioner = pc.partition_str

                # step 2.2.2: sample a synchornizer type for each partition
                parts = []
                for i in range(pc.num_shards):
                    part = strategy_pb2.Strategy.Node()
                    part.var_name = '{}/part_{}:0'.format(get_op_name(var.name), i)
                    self.helpers[part.var_name] = PartHelper(i, var, pc)
                    parts.append(part)
                sample_parts_synchronizers(parts, var_helper, resource_spec, self.space, self.heuristics)
                node.part_config.extend(parts)
            node_config.append(node)

        sample_group_and_reduction_destinations(node_config, resource_spec, self.helpers, self.heuristics)
        # Mark each variable to be synchronized with a Parameter Server
        expr.node_config.extend(node_config)
        return expr


def sample_if_partition(var_helper, resource_spec, space, heuristics):
    reduction_device_names = [k for k, _ in resource_spec.cpu_devices]
    if len(space['maybe_partition']) == 1:
        return space['maybe_partition']
    if heuristics['enable_single_node_no_partition'] and len(reduction_device_names) <= 1:
        return False

    # intersection of variable's partitonable axis and global constraints
    if var_helper.partitionable_axis:
        if space['partitionable_axis']:
            a = set(var_helper.partitionable_axis) & set(space['partitionable_axis'])
            if len(a) < 1:
                return False
    else:
        return False

    # lower bound for abandoning partitioning
    lb = heuristics['maybe_partition_bounds'][0]
    ub = heuristics['maybe_partition_bounds'][1]
    if var_helper.byte_size <= lb:
        return False
    if var_helper.byte_size >= ub:
        return True
    assert (len(space['maybe_partition']) == 2)

    if heuristics['maybe_partition_by_size']:
        #  By variable size -- a large variable has a higher chance to be partitioned
        # TODO (Hao): MAX_INT32 is too large, reconsider later...
        chance = float(var_helper.byte_size - lb) / float(ub - lb)
        return sample_util.binary_sample(boundary=chance)
    else:
        return sample_util.uniform_sample_by_choices(space['maybe_partition'])


def sample_var_synchronizer(node, var_helper, resource_spec, space):
    # sample a single synchornizer for an unpartitioned variable,
    # will eave merge_group of reduction_destination as empty

    # We ALWAYS use PS for sparse variables
    synchronizer_type = 'PS' if var_helper.var_type == VarType.SPARSE \
        else sample_util.uniform_sample_by_choices(space['synchronizer_types'])
    if synchronizer_type == 'PS':
        node.PSSynchronizer.sync = True  # we don't consider async at this moment
        node.PSSynchronizer.staleness = 0
        node.PSSynchronizer.local_replication = sample_if_local_replication(space['local_replication'],
                                                                            resource_spec)
    else:
        # no other option for spec
        node.AllReduceSynchronizer.spec = synchronizers_pb2.AllReduceSynchronizer.Spec.Value('AUTO')
        node.AllReduceSynchronizer.compressor = \
            synchronizers_pb2.AllReduceSynchronizer.Compressor.Value(
                sample_ar_compressor(space['compressor']))


def sample_parts_synchronizers(parts, var_helper, resource_spec, space, heuristics):
    # sample synchornizer for a group of variable partitions

    if var_helper.var_type == VarType.SPARSE:
        synchronizer_types = ['PS'] * len(parts)
    else:
        if heuristics['same_synchronizer_for_parts']:
            type = sample_util.uniform_sample_by_choices(space['synchronizer_types'])
            synchronizer_types = [type] * len(parts)
        else:
            synchronizer_types = [sample_util.uniform_sample_by_choices(space['synchronizer_types'])
                                  for part in parts]
    for i, part in enumerate(parts):
        if synchronizer_types[i] == 'PS':
            part.PSSynchronizer.sync = True  # we don't consider async at this moment
            part.PSSynchronizer.staleness = 0
            part.PSSynchronizer.local_replication = sample_if_local_replication(space['local_replication'],
                                                                                resource_spec)
        else:
            # no other option for spec
            part.AllReduceSynchronizer.spec = synchronizers_pb2.AllReduceSynchronizer.Spec.Value('AUTO')
            part.AllReduceSynchronizer.compressor = \
                synchronizers_pb2.AllReduceSynchronizer.Compressor.Value(
                    sample_ar_compressor(space['compressor']))


def sample_partition_config(var_helper, resource_spec, space, heuristics):
    # Since Arion only support parttion along one axis,
    # we first sample a partition axis, then sammple #partition along that axis, we obtain the partition config.
    assert len(var_helper.partitionable_axis) > 0, 'No partition axis available'
    # sample partition axis
    # TODO(Hao): some heursitics here available?
    valid_axis = var_helper.partitionable_axis
    if space['partitionable_axis']:
        valid_axis = list(set(valid_axis) & set(space['partitionable_axis']))
    partition_axis = sample_util.uniform_sample_by_choices(valid_axis)

    # sample how many partition to go
    num_nodes = resource_spec.num_cpus
    dim_size = var_helper.shape[partition_axis]
    if heuristics['num_partition_bounds'][1] == 'num_nodes':
        max_shards = min(dim_size, num_nodes)
    elif isinstance(heuristics['num_partition_bounds'][1], int):
        max_shards = min(dim_size, heuristics['num_partition_bounds'][1])
    else:
        raise ValueError('unseen num_partition_bounds config')

    min_shards = 2
    if isinstance(heuristics['num_partition_bounds'][0], int):
        min_shards = max(min_shards, heuristics['num_partition_bounds'][0])
    elif heuristics['num_partition_bounds'][0] == 'num_nodes':
        min_shards = max(min_shards, heuristics['num_partition_bounds'][0])
    else:
        raise ValueError('unseen num_partition_bounds config')

    # sample from [min_shards, max_shards]
    num_shards = sample_util.uniform_sample_by_choices(range(min_shards, max_shards + 1))

    # construct a PartitionerConfig (pc)
    partition_list = [1] * len(var_helper.shape)
    partition_list[partition_axis] = num_shards
    pc = PartitionerConfig(partition_list=partition_list)
    return pc


def sample_if_local_replication(local_replication_space, resource_spec):
    # Local replication is a PS-specific semantic; it represents whether to use hierarchical PS
    if resource_spec.num_gpus <= resource_spec.num_cpus:
        # meaning every machine has at most 1 GPU
        return False
    return sample_util.uniform_sample_by_choices(local_replication_space)


def sample_ar_compressor(compressor_space):
    # [NoneCompressor, HorovodCompressor, HorovodCompressorEF, PowerSGDCompressor]
    # [ HorovodCompressorEF, PowerSGDCompressor] will change gradient value
    # so only two choices here
    # TODO(Hao): try to use all four options
    return sample_util.uniform_sample_by_choices(compressor_space)


def sample_group_and_reduction_destinations(node_config, resource_spec, helpers, heuristics):
    ps_shards = OrderedDict()
    ar_shards = OrderedDict()
    idx = 0
    for node in node_config:
        if node.partitioner:
            for part in node.part_config:
                synchronizer = getattr(part, part.WhichOneof('synchronizer'))
                if hasattr(synchronizer, 'compressor'):
                    ar_shards[part.var_name] = (idx,)
                else:
                    ps_shards[part.var_name] = (idx,)
                idx += 1
        else:
            synchronizer = getattr(node, node.WhichOneof('synchronizer'))
            if hasattr(synchronizer, 'compressor'):
                ar_shards[node.var_name] = (idx,)
            else:
                ps_shards[node.var_name] = (idx,)
            idx += 1

    if len(ps_shards) > 0:
        sample_ps_reduction_destinations(node_config, ps_shards, resource_spec, helpers, heuristics)

    # step 4: assign ar merge groups globally
    if len(ar_shards) > 0:
        sample_ar_groups(node_config, ar_shards, helpers, heuristics)


def sample_ps_reduction_destinations(node_config, ps_shards, resource_spec, helpers, heuristics):
    load_balancer = heuristics['ps_load_balancer']
    reduction_device_names = [k for k, _ in resource_spec.cpu_devices]
    if not load_balancer:
        destinations = {}
        for shard_name in ps_shards:
            destinations[shard_name] = sample_util.uniform_sample_by_choices(reduction_device_names)
    elif load_balancer == 'greedy':
        destinations = greedy_load_balancer(ps_shards, resource_spec, helpers)
    elif load_balancer == 'christy':
        # copy Christy's partitionedPS
        destinations = christy_load_balancer(ps_shards, resource_spec, helpers)
    elif load_balancer == 'sorted_christy':
        destinations = christy_load_balancer(ps_shards, resource_spec, helpers, sort_by_size=True)
    elif load_balancer == 'sorted_greedy':
        destinations = greedy_load_balancer(ps_shards, resource_spec, helpers, sort_by_size=True)
    else:
        raise ValueError('Cannot recognize load balancer')

    for shard_name, (idx, ) in ps_shards.items():
        ps_shards[shard_name] = (idx, destinations[shard_name])

    assign_ps_reduction_destinations(node_config, ps_shards)


def assign_ps_reduction_destinations(node_config, ps_shards):
    for node in node_config:
        if node.partitioner:
            for part in node.part_config:
                synchronizer = getattr(part, part.WhichOneof('synchronizer'))
                if hasattr(synchronizer, 'reduction_destination'):
                    synchronizer.reduction_destination = ps_shards[part.var_name][1]
        else:
            synchronizer = getattr(node, node.WhichOneof('synchronizer'))
            if hasattr(synchronizer, 'reduction_destination'):
                synchronizer.reduction_destination = ps_shards[node.var_name][1]


def sample_ar_groups(node_config, ar_shards, helpers, heuristics):
    merge_scheme = heuristics['merge_scheme']
    if merge_scheme == 'by_chunk':
        if 'chunk_size' in heuristics and heuristics['chunk_size'] > 0:
            chunk_size_or_num_group = heuristics['chunk_size']
        else:
            chunk_size_or_num_group = sample_chunk_size(len(ar_shards))
    else:
        chunk_size_or_num_group = sample_num_ar_groups(ar_shards,
                                                       heuristics['num_group_bounds'][0],
                                                       heuristics['num_group_bounds'][1])
    assert chunk_size_or_num_group > 0, "chunk_size or num_groups need to > 1..."

    if merge_scheme in ['random', None]:
        tmp_assignments = sample_util.sample_merge_group(chunk_size_or_num_group, len(ar_shards))
        group_assignments = OrderedDict()
        for i, shard_name in enumerate(ar_shards):
            group_assignments[shard_name] = tmp_assignments[i]
    elif merge_scheme == 'by_chunk':
        # sample chunk_size
        group_assignments = chunk_group_assigner(ar_shards, chunk_size_or_num_group)
    elif merge_scheme == 'christy':
        group_assignments = christy_group_assigner(ar_shards,
                                                   helpers,
                                                   chunk_size_or_num_group)
    elif merge_scheme == 'ordered_balanced':
        group_assignments = ordered_balanced_group_assigner(ar_shards,
                                                            helpers,
                                                            chunk_size_or_num_group)
    else:
        raise ValueError('unseen merge scheme..')

    for shard_name, (idx,) in ar_shards.items():
        ar_shards[shard_name] = (idx, group_assignments[shard_name])
    assign_ar_group(node_config, ar_shards)


def sample_num_ar_groups(ar_shards, lb, ub):
    min_num_group = max(1, lb)
    max_num_group = min(len(ar_shards), ub)
    num_group = sample_util.uniform_sample_by_choices(range(min_num_group, max_num_group + 1))
    return num_group


def sample_chunk_size(num_ar_shards):
    chunk_size = sample_util.uniform_sample_by_choices(range(1, num_ar_shards + 1))
    return chunk_size


def assign_ar_group(node_config, ar_shards):
    for node in node_config:
        if node.partitioner:
            for part in node.part_config:
                synchronizer = getattr(part, part.WhichOneof('synchronizer'))
                if hasattr(synchronizer, 'compressor'):
                    synchronizer.group = ar_shards[part.var_name][1]
        else:
            synchronizer = getattr(node, node.WhichOneof('synchronizer'))
            if hasattr(synchronizer, 'compressor'):
                synchronizer.group = ar_shards[node.var_name][1]
