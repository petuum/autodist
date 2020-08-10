"""Strategy RankNetSimulator."""
import glob
import json
import sys
from datetime import datetime
from pathlib import Path
from string import digits
import time

import numpy as np
import os
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import arion
from arion.graph_item import GraphItem
from arion.proto.synchronizers_pb2 import PSSynchronizer, AllReduceSynchronizer
from arion.simulator.models.base import SimulatorBase
from arion.simulator.utils import get_dense_var_bits, get_sparse_var_bits, GIGABITS
from arion.simulator.utils import _resolve_device_address, _max_num_local_replica, _num_local_replica, _resolved_devices_on_diff_machine
from arion.strategy.random_sample_strategy import VariableHelper, PartHelper
from arion.strategy.base import Strategy
from arion.resource_spec import ResourceSpec
from arion.cluster import SSHCluster
from arion.kernel.device.resolver import DeviceResolver
from arion.kernel.partitioner import PartitionerConfig
from arion.simulator.models.predefined_simulator import PredefinedSimulator

import torch
import torch.nn as nn

import multiprocessing
from multiprocessing import Process, Queue

TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# feature settings
MAX_NUM_WORKERS = 16
MAX_NUM_GROUPS = 600
MAX_NUM_VARS = 500
MAX_NUM_PARS = 1500
FEATURE_SIZE = MAX_NUM_WORKERS+MAX_NUM_GROUPS+15

# model size
PARTITION_MLP_HIDDEN = 128
PARTITION_MLP_OUT = 32
STEM_RNN_HIDDEN = 128
BIDIECTIONAL = True
BATCH_SIZE = 96

NUM_RNN_LAYERS = 3
SCORE_TH = 0.005
LR = 2e-3
WD = 3e-4
DATA_AUG = False
IN_LAYERS = 2
OUT_LAYERS = 1

# ncf used:
# ~/projects/pycharm/zhijie/5-9-2020/oceanus-zhijie/arion/simulator/models/model_train_on_ncf-orca_new.ckpt 0.9020
# noaug
# PARTITION_MLP_HIDDEN = 128
# PARTITION_MLP_OUT = 32
# STEM_RNN_HIDDEN = 128
# BIDIECTIONAL = True
# NUM_RNN_LAYERS = 4
# BATCH_SIZE = 64
# LR = 1e-3
# WD = 4e-4

# vgg used:
# ~/projects/pycharm/zhijie/5-9-2020/oceanus-zhijie/arion/simulator/models/model_train_on_vgg16-orca_new_new_new.ckpt 0.8374
# noaug
# PARTITION_MLP_HIDDEN = 128
# PARTITION_MLP_OUT = 32
# STEM_RNN_HIDDEN = 128
# BIDIECTIONAL = True
# NUM_RNN_LAYERS = 3
# BATCH_SIZE = 64
# LR = 1e-3
# WD = 3e-4

GRAPH_ITEM_PATHS = {'ncf':'/users/hzhang2/projects/pycharm/zhijie/5-5-2020/original_graph_item',
                'densenet121': '/users/hzhang2/projects/pycharm/zhijie/graph_items/densenet121_original_graph_item',
                'inceptionv3': '/users/hzhang2/projects/pycharm/zhijie/graph_items/inceptionv3_original_graph_item',
                'resnet101': '/users/hzhang2/projects/pycharm/zhijie/graph_items/resnet101_original_graph_item',
                'resnet50': '/users/hzhang2/projects/pycharm/zhijie/graph_items/resnet50_original_graph_item',
                'vgg16': '/users/hzhang2/projects/pycharm/zhijie/graph_items/vgg16_original_graph_item',
                'bert_12l': '/users/hzhang2/projects/pycharm/zhijie/graph_items/bert_original_graph_item_12l',
                'bert_6l': '/users/hzhang2/projects/pycharm/zhijie/graph_items/bert_original_graph_item_6l',
                'bert_3l': '/users/hzhang2/projects/pycharm/zhijie/graph_items/bert_original_graph_item_3l',
                'bert_large': '/users/hzhang2/projects/pycharm/zhijie/graph_items/bert_original_graph_item_large'}

def get_model(path_):
    if 'densenet121' in path_:
        return 'densenet121'
    elif 'ncf' in path_:
        return 'ncf'
    elif 'inceptionv3' in path_:
        return 'inceptionv3'
    elif 'resnet101' in path_:
        return 'resnet101'
    elif 'resnet50' in path_:
        return 'resnet50'
    elif 'vgg16' in path_:
        return 'vgg16'
    elif 'bert' in path_ and '12l' in path_:
        return 'bert_12l'
    elif 'bert' in path_ and '6l' in path_:
        return 'bert_6l'
    elif 'bert' in path_ and '3l' in path_:
        return 'bert_3l'
    elif 'bert' in path_ and 'large' in path_:
        return 'bert_large'
    else:
        return None

class RankRNN(nn.Module):
    def __init__(self, input_size=FEATURE_SIZE,
                       partition_mlp_hidden=PARTITION_MLP_HIDDEN, 
                       partition_mlp_out=PARTITION_MLP_OUT, 
                       stem_rnn_hidden=STEM_RNN_HIDDEN, 
                       num_rnn_layers=NUM_RNN_LAYERS, 
                       in_layers=IN_LAYERS,
                       out_layers=OUT_LAYERS,
                       bidirectional=BIDIECTIONAL):
        super(RankRNN, self).__init__()
        self.partition_mlp_out = partition_mlp_out
        # self.num_rnn_layers = num_rnn_layers
        self.stem_rnn_hidden = stem_rnn_hidden
        tmp = [nn.Linear(input_size, partition_mlp_hidden)]
        for _ in range(in_layers-2):
            tmp.append(nn.ReLU())
            tmp.append(nn.Linear(partition_mlp_hidden, partition_mlp_hidden))
        tmp.append(nn.ReLU())
        tmp.append(nn.Linear(partition_mlp_hidden, partition_mlp_out))

        self.partition_mlp = nn.Sequential(*tmp)

        self.stem_rnn = nn.LSTM(partition_mlp_out, stem_rnn_hidden, num_rnn_layers, batch_first=True, bidirectional=bidirectional)

        if out_layers == 1:
            self.final_fc = nn.Linear(stem_rnn_hidden*num_rnn_layers*(1+int(bidirectional)), 1)
        elif out_layers == 2:
            self.final_fc = nn.Sequential(nn.Linear(stem_rnn_hidden*num_rnn_layers*(1+int(bidirectional)), 128),
                                          nn.ReLU(),
                                          nn.Linear(128, 1))

        self.relu = nn.ReLU()
    
    def forward(self, features, par_indices, var_nums, return_feature=False):
        # print(features.shape, par_indices.shape, var_nums.shape)
        x = features.float()
        # x = torch.cat([features[:, :, :MAX_NUM_WORKERS], features[:, :, MAX_NUM_WORKERS+MAX_NUM_GROUPS:]], 2).float()
        x = self.partition_mlp(x)

        x1 = torch.zeros(features.shape[0], MAX_NUM_VARS, self.partition_mlp_out, device=TORCH_DEVICE, dtype=x.dtype)
        x1.scatter_add_(1, par_indices.long()[:, :, None].expand(par_indices.shape[0], par_indices.shape[1], self.partition_mlp_out), x)

        # Set initial hidden and cell states 
        # h0 = torch.zeros(self.num_rnn_layers, x.size(0), self.stem_rnn_hidden).to(TORCH_DEVICE) 
        # c0 = torch.zeros(self.num_rnn_layers, x.size(0), self.stem_rnn_hidden).to(TORCH_DEVICE)
        
        # Forward propagate LSTM
        x1 = torch.nn.utils.rnn.pack_padded_sequence(x1, var_nums.long(), batch_first=True, enforce_sorted=False)
        out, (ht, ct) = self.stem_rnn(x1)  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # out = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)[0].sum(1) / var_nums[:, None]
        out = ht.permute(1, 0, 2).reshape(x.shape[0], -1)
        # print(out[0, var_nums[0] -1, [3]], out[0, var_nums[0], [3]])
        # print(ht.permute(1, 0, 2).shape, x.shape)
        if return_feature:
            return self.final_fc(out), out.div((out**2).sum(1, keepdim=True).sqrt())
        else:
            return self.final_fc(out)

class TrainTensorDataset(torch.utils.data.Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        x = self.tensors[0][index]
        x = self.perturbe_device_and_group(x)
        x1 = self.tensors[1][index]
        x2 = self.tensors[2][index]

        y = self.tensors[3][index]

        return x, x1, x2, y

    def __len__(self):
        return self.tensors[0].size(0)

    def perturbe_device_and_group(self, x):
        if DATA_AUG:
            perturbed_device_ids = np.random.permutation(MAX_NUM_WORKERS).astype(np.int32)
            perturbed_group_ids = np.random.permutation(MAX_NUM_GROUPS).astype(np.int32)
            mat_device = torch.eye(MAX_NUM_WORKERS, device=x.device, dtype=x.dtype)[perturbed_device_ids]
            mat_group = torch.eye(MAX_NUM_GROUPS, device=x.device, dtype=x.dtype)[perturbed_group_ids]
            x = torch.cat([torch.matmul(x[:, :MAX_NUM_WORKERS], mat_device), torch.matmul(x[:, MAX_NUM_WORKERS:MAX_NUM_WORKERS+MAX_NUM_GROUPS], mat_group), x[:, MAX_NUM_WORKERS+MAX_NUM_GROUPS:]], 1)
        return x


def to_numpy(synchronizer, device, size_ratio, is_sparse, bd, num_replicas):
    ret = [np.zeros(MAX_NUM_WORKERS), np.zeros(MAX_NUM_GROUPS), np.zeros(3), np.zeros(5), np.zeros(3)]

    if device is not None:
        ret[0][device] = 1
    
    group = getattr(synchronizer, 'group', None)
    if group is not None:
        assert group < MAX_NUM_GROUPS, group
        ret[1][group] = 1

    compressor = getattr(synchronizer, 'compressor', None)
    if compressor is not None:
        if compressor in ["PowerSGDCompressor", 3]:
            ret[2][2] = 1
        elif compressor in ["HorovodCompressorEF", "HorovodCompressor", 2, 1]:
            ret[2][1] = 1
        elif compressor in ["NoneCompressor", 0]:
            ret[2][0] = 1
        else:
            raise ValueError('Compressor does not exist: {}'.format(compressor))

    local_replication = getattr(synchronizer, 'local_replication', None)
    if isinstance(synchronizer, PSSynchronizer):
        synchronizer = 0
        if int(local_replication) == 0:
            if int(is_sparse) == 0:
                ret[3][0] = 1
            else:
                ret[3][1] = 1
        else:
            if int(is_sparse) == 0:
                ret[3][2] = 1
            else:
                ret[3][3] = 1
    else:
        ret[3][4] = 1
    ret[4] = np.array([size_ratio, bd, num_replicas])

    return np.concatenate(ret)

def connvert_feature(strategy, resource_spec, graph_item):
    
    cluster = SSHCluster(resource_spec)
    device_resolver = DeviceResolver(cluster)
    graph_replicas = [_resolve_device_address(k, device_resolver) for k, v in resource_spec.gpu_devices]
    # bandwidth
    network_bandwidth = np.array([resource_spec.network_bandwidth[device.split(':')[0]] for device, _ in resource_spec.cpu_devices])
    network_bandwidth = network_bandwidth
    min_network_bandwidth = network_bandwidth.min()
    # Other information
    cpu_worker_list = [_resolve_device_address(device, device_resolver) for device, _ in resource_spec.cpu_devices]
    gpu_worker_list = [_resolve_device_address(device, device_resolver) for device, _ in resource_spec.gpu_devices]
    max_num_local_replica = _max_num_local_replica(graph_replicas, cluster)
    total_num_local_replica = len(graph_replicas)
    worker_num_replicas = [_num_local_replica(cpu_worker, graph_replicas, cluster) for cpu_worker in cpu_worker_list]

    num_vars = 0
    total_size_vars = 0
    for var_op, var in graph_item.trainable_var_op_to_var.items():
        num_vars += 1
        if var.initial_value.shape.ndims:
            var_helper = VariableHelper(var, graph_item)
            if var_helper.is_sparse:
                total_size_vars += get_sparse_var_bits(np.prod(var_helper.shape))
            else:
                total_size_vars += get_dense_var_bits(np.prod(var_helper.shape), var.dtype)
    assert num_vars < MAX_NUM_VARS, num_vars
    var_partition_features = np.zeros((MAX_NUM_PARS, FEATURE_SIZE-4)).astype(np.float32)
    partition_indice = np.ones(MAX_NUM_PARS).astype(np.float32) * (MAX_NUM_VARS - 1)

    cnt = 0
    for node_id, node in enumerate(strategy.node_config):
        var_name = node.var_name
        for var_op, var in graph_item.trainable_var_op_to_var.items():
            if var.name == var_name:
                break
        var_helper = VariableHelper(var, graph_item)

        if node.partitioner:
            pc = PartitionerConfig(partition_str=node.partitioner)
            for i, part in enumerate(node.part_config):
                part_helper = PartHelper(i, var, pc)
                synchronizer = getattr(part, part.WhichOneof('synchronizer'))
                reduction_destination = getattr(synchronizer, 'reduction_destination', None)
                device = _resolve_device_address(reduction_destination if reduction_destination else var.device,
                                                 device_resolver)
                if device == '':
                    assert(isinstance(synchronizer, AllReduceSynchronizer))
                    device = None
                    bd = min_network_bandwidth
                    num_replicas = 0
                else:
                    device = cpu_worker_list.index(device)
                    bd = network_bandwidth[device]
                    num_replicas = worker_num_replicas[device]

                if var_helper.is_sparse:
                    size_ratio = get_sparse_var_bits(np.prod(part_helper.shape))/total_size_vars
                else:
                    size_ratio = get_dense_var_bits(np.prod(part_helper.shape), var_helper.dtype)/total_size_vars
                var_partition_features[cnt] = to_numpy(synchronizer, device, size_ratio, var_helper.is_sparse, bd, num_replicas)
                partition_indice[cnt] = node_id
                cnt += 1
        else:
            synchronizer = getattr(node, node.WhichOneof('synchronizer'))
            reduction_destination = getattr(synchronizer, 'reduction_destination', None)
            device = _resolve_device_address(reduction_destination if reduction_destination else var.device,
                                             device_resolver)
            if device == '':
                assert(isinstance(synchronizer, AllReduceSynchronizer))
                device = None
                bd = min_network_bandwidth
                num_replicas = 0
            else:
                device = cpu_worker_list.index(device)
                bd = network_bandwidth[device]
                num_replicas = worker_num_replicas[device]

            if var_helper.is_sparse:
                size_ratio = get_sparse_var_bits(np.prod(var_helper.shape))/total_size_vars
            else:
                size_ratio = get_dense_var_bits(np.prod(var_helper.shape), var_helper.dtype)/total_size_vars
            var_partition_features[cnt] = to_numpy(synchronizer, device, size_ratio, var_helper.is_sparse, bd, num_replicas)
            partition_indice[cnt] = node_id
            cnt += 1
    return var_partition_features, partition_indice, np.array(node_id+1)

def create_predefined_features(strategy, resource_spec, predefined_simulator):

    var_sync_time, vars, resource = predefined_simulator.predefined_sync_time(strategy, resource_spec)

    features = []
    for var_name, sync_time in var_sync_time.items():
        if isinstance(sync_time, list) or isinstance(sync_time, tuple): # send_time, receive_time in PS strategies.
            transmission = sync_time[0]['transmission'] + sync_time[1]['transmission']
            sync_time = sync_time[0]
            is_ps = True
        else:   # AR
            transmission = sync_time['transmission']
            is_ps = False

        network_overhead = sync_time['network_overhead']
        gpu_kernel_memory_latency = sync_time['gpu_kernel_memory_latency']

        feat = [transmission, network_overhead, gpu_kernel_memory_latency, float(is_ps)]
        features.append(feat)
    features = np.array(features, dtype=np.float)
    return features

def extract_graph_item(graph_item):
    total_size_vars = 0
    name2var = {}
    name2var_helper = {}
    for var_op, var in graph_item.trainable_var_op_to_var.items():
        name2var[var.name] = var
        var_helper = VariableHelper(var, graph_item)
        name2var_helper[var.name] = var_helper
        if var.initial_value.shape.ndims:
            if var_helper.is_sparse:
                total_size_vars += get_sparse_var_bits(np.prod(var_helper.shape))
            else:
                total_size_vars += get_dense_var_bits(np.prod(var_helper.shape), var.dtype)

    return total_size_vars, name2var, name2var_helper

def wrap_fn(queue, idx, run_worker, rs, st):
    ret = run_worker(rs, st)
    queue.put((idx, ret))

def convert_feature_batch(strategys, resource_specs, total_size_vars, name2var, name2var_helper, _batch_size_per_gpu, _seq_len):

    def var_ps_time(var_size_to_transfer, is_sparse, device, dtype, local_replication, network_bandwidth_map, max_num_local_replica, cpu_worker_list, gpu_worker_list,  network_overhead=0.0, gpu_kernel_memory_latency=0.0):
        """Compute synchronization time of a variable in PS strategy."""
        def _helper(worker_list, worker_num_replicas=None):
            if worker_num_replicas is None:
                worker_num_replicas = [1.0] * len(worker_list)

            this_server_time = 0
            # network transfer: sum up all workers time. equals to the time cost of this server.
            # TODO(Hao): didn't consider any parallelization among partitions
            for k, worker in enumerate(worker_list):
                if _resolved_devices_on_diff_machine(device, worker):
                    if is_sparse:
                        this_worker_size = get_sparse_var_bits(var_size_to_transfer) * worker_num_replicas[k]
                    else:
                        this_worker_size = get_dense_var_bits(var_size_to_transfer, dtype)
                    this_server_time += this_worker_size / network_bandwidth_map[device][worker]

            return {
                'transmission': this_server_time,
                'network_overhead': len(worker_list),
                'gpu_kernel_memory_latency': max_num_local_replica,
            }

        send_time = _helper(cpu_worker_list)
        if local_replication:
            receive_time = _helper(cpu_worker_list)
        else:
            receive_time = _helper(gpu_worker_list)

        return send_time, receive_time

    def var_ar_time(var_size_to_transfer, og_shape, dtype, compressor, max_num_local_replica, cpu_worker_list, network_bandwidth_map, network_overhead=0.0, gpu_kernel_memory_latency=0.0):
        """Compute synchronization time of a variable in AR strategy."""
        worker_list = cpu_worker_list
        num_workers = len(worker_list)
        min_bandwidth = None
        for i in range(num_workers):
            for j in range(i, num_workers):
                if min_bandwidth is None:
                    min_bandwidth = network_bandwidth_map[worker_list[j]][worker_list[i]]
                else:
                    min_bandwidth = min(min_bandwidth, network_bandwidth_map[worker_list[j]][worker_list[i]])

        # Compressor
        if compressor == "PowerSGDCompressor" or compressor == 3:
            rank = 10  # currently using default value. So hardcode here. # todo: confirm
            # assume var must be a dense variable.
            ndims = len(og_shape)
            if ndims <= 1:  # no compress
                size_to_transfer = var_size_to_transfer
            else:
                if ndims > 2:
                    n = og_shape[0]
                    m = 1
                    for s in og_shape[1:]:
                        m *= s  # tensor's shape (n, m)
                else:
                    n, m = og_shape[0], og_shape[1]
                size_to_transfer = n * rank + m * rank
            dtype = tf.float32
        elif compressor == "HorovodCompressorEF" or compressor == "HorovodCompressor"  \
                or compressor == 2  or compressor == 1:
            size_to_transfer = var_size_to_transfer
            dtype = tf.float32
        elif compressor == "NoneCompressor" or compressor == 0:
            size_to_transfer = var_size_to_transfer
            dtype = dtype
        else:
            raise ValueError('Compressor does not exist: {}'.format(compressor))

        time = get_dense_var_bits(size_to_transfer, dtype) / min_bandwidth

        return {
            'transmission': time,
            'network_overhead': 1,  # len(worker_list),
            'gpu_kernel_memory_latency': max_num_local_replica,
        }

    def network_bandwidth2(resource_spec: ResourceSpec, device_resolver: DeviceResolver):
        """Calculates all P2P network bandwidths between nodes in the cluster."""
        devices = [device for device, _ in resource_spec.devices]
        resolved_devices = [_resolve_device_address(device, device_resolver) for device, _ in resource_spec.devices]
        gpu_cpu_bw = 10000.  # hardcode for now
        network_bandwidth = {}  # key: <server, worker>
        for i in range(len(devices)):
            if resolved_devices[i] not in network_bandwidth:
                network_bandwidth[resolved_devices[i]] = {}
            for j in range(i, len(devices)):
                if resolved_devices[j] not in network_bandwidth:
                    network_bandwidth[resolved_devices[j]] = {}
                ip_i = devices[i].split(':')[0]
                ip_j = devices[j].split(':')[0]
                if ip_i != ip_j:
                    network_bandwidth[resolved_devices[i]][resolved_devices[j]] \
                        = GIGABITS * resource_spec.network_bandwidth[ip_i]
                    network_bandwidth[resolved_devices[j]][resolved_devices[i]] \
                        = GIGABITS * resource_spec.network_bandwidth[ip_j]
                else:
                    network_bandwidth[resolved_devices[i]][resolved_devices[j]] = GIGABITS * gpu_cpu_bw
                    network_bandwidth[resolved_devices[j]][resolved_devices[i]] = GIGABITS * gpu_cpu_bw
        return network_bandwidth

    def run_worker(resource_spec, strategy):
        cluster = SSHCluster(resource_spec)
        device_resolver = DeviceResolver(cluster)
        graph_replicas = [_resolve_device_address(k, device_resolver) for k, v in resource_spec.gpu_devices]
        # bandwidth
        network_bandwidth = np.array([resource_spec.network_bandwidth[device.split(':')[0]] for device, _ in resource_spec.cpu_devices])
        min_network_bandwidth = network_bandwidth.min()
        network_bandwidth_map = network_bandwidth2(resource_spec, device_resolver)
        # Other information
        cpu_worker_list = [_resolve_device_address(device, device_resolver) for device, _ in resource_spec.cpu_devices]
        gpu_worker_list = [_resolve_device_address(device, device_resolver) for device, _ in resource_spec.gpu_devices]
        max_num_local_replica = _max_num_local_replica(graph_replicas, cluster)
        total_num_local_replica = len(graph_replicas)
        worker_num_replicas = [_num_local_replica(cpu_worker, graph_replicas, cluster) for cpu_worker in cpu_worker_list]

        var_partition_features = np.zeros((MAX_NUM_PARS, FEATURE_SIZE)).astype(np.float32)
        partition_indice = np.ones(MAX_NUM_PARS).astype(np.float32) * (MAX_NUM_VARS - 1)
        cnt = 0
        for node_id, node in enumerate(strategy.node_config):
            var_name = node.var_name
            var = name2var[var_name]
            var_helper = name2var_helper[var_name]

            if node.partitioner:
                pc = PartitionerConfig(partition_str=node.partitioner)
                for i, part in enumerate(node.part_config):
                    synchronizer = getattr(part, part.WhichOneof('synchronizer'))
                    reduction_destination = getattr(synchronizer, 'reduction_destination', None)
                    device = _resolve_device_address(reduction_destination if reduction_destination else var.device,
                                                     device_resolver)
                    if device == '':
                        assert(isinstance(synchronizer, AllReduceSynchronizer))
                        device_id = None
                        bd = min_network_bandwidth
                        num_replicas = 0
                    else:
                        device_id = cpu_worker_list.index(device)
                        bd = network_bandwidth[device_id]
                        num_replicas = worker_num_replicas[device_id]

                    par_shape = var.initial_value.shape.as_list()
                    dim_size = par_shape[pc.axis] // pc.num_shards
                    extras = par_shape[pc.axis] % pc.num_shards
                    if i < extras:
                        dim_size += 1
                    par_shape[pc.axis] = dim_size

                    size_to_transfer =np.prod(par_shape)
                    if var_helper.is_sparse:
                        raise Error
                        size_ratio = get_sparse_var_bits(size_to_transfer)/total_size_vars
                    else:
                        size_ratio = get_dense_var_bits(size_to_transfer, var_helper.dtype)/total_size_vars

                    if isinstance(synchronizer, AllReduceSynchronizer):
                        sync_time = var_ar_time(size_to_transfer, par_shape, var_helper.dtype, getattr(synchronizer, 'compressor', None), max_num_local_replica, cpu_worker_list, network_bandwidth_map)
                        transmission = sync_time['transmission']
                        is_ps = False
                    else:
                        sync_time = var_ps_time(size_to_transfer, var_helper.is_sparse, device, var_helper.dtype, getattr(synchronizer, 'local_replication', None), network_bandwidth_map, max_num_local_replica, cpu_worker_list, gpu_worker_list)
                        transmission = sync_time[0]['transmission'] + sync_time[1]['transmission']
                        sync_time = sync_time[0]
                        is_ps = True
                    network_overhead = sync_time['network_overhead']
                    gpu_kernel_memory_latency = sync_time['gpu_kernel_memory_latency']

                    var_partition_features[cnt] = np.concatenate([to_numpy(synchronizer, device_id, size_ratio, var_helper.is_sparse, bd, num_replicas), np.array([transmission, network_overhead, gpu_kernel_memory_latency, float(is_ps)])])
                    partition_indice[cnt] = node_id
                    cnt += 1
            else:
                synchronizer = getattr(node, node.WhichOneof('synchronizer'))
                reduction_destination = getattr(synchronizer, 'reduction_destination', None)
                device = _resolve_device_address(reduction_destination if reduction_destination else var.device,
                                                 device_resolver)
                if device == '':
                    assert(isinstance(synchronizer, AllReduceSynchronizer))
                    device_id = None
                    bd = min_network_bandwidth
                    num_replicas = 0
                else:
                    device_id = cpu_worker_list.index(device)
                    bd = network_bandwidth[device_id]
                    num_replicas = worker_num_replicas[device_id]

                size_to_transfer =np.prod(var_helper.shape)
                if var_helper.is_sparse:
                    raise Error
                    size_ratio = get_sparse_var_bits(size_to_transfer)/total_size_vars
                else:
                    size_ratio = get_dense_var_bits(size_to_transfer, var_helper.dtype)/total_size_vars

                if isinstance(synchronizer, AllReduceSynchronizer):
                    sync_time = var_ar_time(size_to_transfer, var.initial_value.shape.as_list(), var_helper.dtype, getattr(synchronizer, 'compressor', None), max_num_local_replica, cpu_worker_list, network_bandwidth_map)
                    transmission = sync_time['transmission']
                    is_ps = False
                else:
                    sync_time = var_ps_time(size_to_transfer, var_helper.is_sparse, device, var_helper.dtype, getattr(synchronizer, 'local_replication', None), network_bandwidth_map, max_num_local_replica, cpu_worker_list, gpu_worker_list)
                    transmission = sync_time[0]['transmission'] + sync_time[1]['transmission']
                    sync_time = sync_time[0]
                    is_ps = True
                network_overhead = sync_time['network_overhead']
                gpu_kernel_memory_latency = sync_time['gpu_kernel_memory_latency']

                var_partition_features[cnt] = np.concatenate([to_numpy(synchronizer, device_id, size_ratio, var_helper.is_sparse, bd, num_replicas), np.array([transmission, network_overhead, gpu_kernel_memory_latency, float(is_ps)])])
                partition_indice[cnt] = node_id
                cnt += 1
        return (var_partition_features, partition_indice, np.array(node_id+1))

    # t1 =time.time()
    # with multiprocessing.Pool(processes=32) as pool:
    #     results = pool.starmap(run_worker, zip(resource_specs, strategys))
    # ret1, ret2, ret3 = [], [], []        
    # for tmp in results:
    #     ret1.append(tmp[0]); ret2.append(tmp[1]); ret3.append(tmp[2])

    q = Queue()
    rets = []
    prs = []
    for idx, (arg1, arg2) in enumerate(zip(resource_specs, strategys)):
        prs.append(Process(target=wrap_fn, args=(q, idx, run_worker, arg1, arg2)))
        prs[-1].start()
    for pr in prs:
        ret = q.get() # will block
        rets.append(ret)
    for pr in prs:
        pr.join()

    ret1, ret2, ret3 = [], [], []
    for tmp in sorted(rets, key=lambda x: x[0]):
        ret1.append(tmp[1][0]); ret2.append(tmp[1][1]); ret3.append(tmp[1][2])
    # print(time.time() - t1)

    # t1 =time.time()
    # ret1, ret2, ret3 = [], [], []
    # for rs, st in zip(resource_specs, strategys):
    #     tmp = run_worker(rs, st)
    #     ret1.append(tmp[0]); ret2.append(tmp[1]); ret3.append(tmp[2])
    # print(time.time() - t1)
    return np.stack(ret1), np.stack(ret2), np.stack(ret3)


class RankRNNSimulatorPenalty(SimulatorBase):
    """Simulates strategies for a given graph and resource spec."""

    def __init__(self,
                 original_graph_item_path,
                 num_rnn_layers,
                 in_layers,
                 out_layers,
                 fetches=None,
                 batch_size=1,
                 seq_len=1,
                 checkpoint=None):

        super(RankRNNSimulatorPenalty, self).__init__(original_graph_item_path=original_graph_item_path)
        print("It's using RankNet simulator.")
        self._fetches = fetches
        self._batch_size_per_gpu = batch_size
        self._seq_len = seq_len
        self._checkpoint = checkpoint
        self._predefined_simulator=PredefinedSimulator(original_graph_item_path=original_graph_item_path,
                                                         batch_size=self._batch_size_per_gpu,
                                                         seq_len=self._seq_len)
        if self._checkpoint:
            self._model = RankRNN(num_rnn_layers=num_rnn_layers, in_layers=in_layers, out_layers=out_layers).to(TORCH_DEVICE)
            self._model.load_state_dict(torch.load(self._checkpoint, map_location=torch.device('cpu')))

        total_size_vars, name2var, name2var_helper = extract_graph_item(self._original_graph_item)
        self.total_size_vars = total_size_vars
        self.name2var = name2var
        self.name2var_helper = name2var_helper

    def simulate(self, strategy, resource_spec, strategy_path=None, checkpoint=None):
        score, feature = self.predict(strategy, resource_spec, strategy_path, checkpoint)
        return score.view(-1).data.cpu().numpy(), feature.data.cpu().numpy()

    def predict(self,
                strategy,
                resource_spec,
                strategy_path=None,
                checkpoint=None):
        if checkpoint is None:
            if self._checkpoint is None:
                raise ValueError("checkpoint is None: {}".format(checkpoint))
            else:
                model = self._model
        else:
            model = RankRNN().to(TORCH_DEVICE)
            model.load_state_dict(torch.load(checkpoint))
        if type(strategy) == list and type(resource_spec) == list:
            
            var_partition_features, partition_indice, var_num = convert_feature_batch(strategy, resource_spec, self.total_size_vars, self.name2var, self.name2var_helper, self._batch_size_per_gpu, self._seq_len)

            var_partition_features = torch.from_numpy(var_partition_features).to(TORCH_DEVICE)
            partition_indice = torch.from_numpy(partition_indice).to(TORCH_DEVICE)
            var_num = torch.from_numpy(var_num).to(TORCH_DEVICE)

            return model(var_partition_features, partition_indice, var_num, True)
        else:
            if strategy_path and os.path.isfile((strategy_path+'.npz').replace('strategies', 'npz')):
                loaded = np.load((strategy_path+'.npz').replace('strategies', 'npz'))
                var_partition_features, partition_indice, var_num, _ = \
                                loaded['x1'], loaded['x2'], loaded['x3'], loaded['y']
            else:
                var_partition_features, partition_indice, var_num = \
                                connvert_feature(strategy, resource_spec, self._original_graph_item)

            if strategy_path and os.path.isfile((strategy_path+'_pdf.npz').replace('strategies', 'npz')):
                loaded = np.load((strategy_path+'_pdf.npz').replace('strategies', 'npz'))
                predefined_features = loaded['x4']
            else:
                predefined_features = create_predefined_features(strategy, resource_spec, self._predefined_simulator)
                
            var_partition_features = np.concatenate([var_partition_features, np.concatenate([predefined_features, np.zeros((MAX_NUM_PARS-predefined_features.shape[0], predefined_features.shape[1]))], 0)], 1)

            var_partition_features = torch.from_numpy(var_partition_features).unsqueeze(0).to(TORCH_DEVICE)
            partition_indice = torch.from_numpy(partition_indice).unsqueeze(0).to(TORCH_DEVICE)
            var_num = torch.from_numpy(var_num).unsqueeze(0).to(TORCH_DEVICE)

            return model(var_partition_features, partition_indice, var_num, True)

class RankNetTrainer():

    def __init__(self,
                 batch_size_per_gpu=256,
                 seq_len=1,
                 seed=1):
        self._batch_size_per_gpu = batch_size_per_gpu
        self._seq_len = seq_len
        self.graph_items = {k:GraphItem.deserialize(v) for k, v in GRAPH_ITEM_PATHS.items()}
        self.predefined_simulators = {k: PredefinedSimulator(original_graph_item_path=v,
                                                         batch_size=self._batch_size_per_gpu,
                                                         seq_len=self._seq_len) for k, v in GRAPH_ITEM_PATHS.items()}
        self.best_acc = 0.
        print("It's using RankNet trainer.")

    def load_data(self, path_list, train_patterns=[('ncf', 0)], valid_patterns='same'):
        features = {k: [[[], [], [], []], [[], [], [], []]] for k, _ in GRAPH_ITEM_PATHS.items()}
        for training_path in path_list:
            for path in Path(training_path).rglob('strategies'):
                strategy_paths = glob.glob(os.path.join(path, '*'))
                # strategy_paths = np.random.permutation(list(strategy_paths))
                for strategy_path in strategy_paths:
                    if 'json' in strategy_path or \
                      'bert_large_batch_8_orca_16_group_2/' in strategy_path:
                        continue
                    model = get_model(strategy_path)
                    if model is None:
                        if not ('densenets169' in strategy_path or 'densenets201' in strategy_path):
                            assert False, strategy_path
                        continue
                    rs_path = strategy_path.replace('strategies', 'resource_specs')
                    runtime_path = strategy_path.replace('strategies', 'runtimes')
                    npz_path = (strategy_path+'.npz').replace('strategies', 'npz')
                    if not os.path.isfile(rs_path):
                        rs_path += '.yml'  
                    if not (os.path.isfile(rs_path) and os.path.isfile(runtime_path)):
                        continue
                    if not os.path.exists(os.path.dirname(npz_path)):
                        os.makedirs(os.path.dirname(npz_path))

                    if not os.path.isfile(npz_path):
                        strategy = Strategy.deserialize(path=strategy_path)
                        rs = ResourceSpec(resource_file=rs_path)
                        var_partition_features, partition_indice, var_num = \
                                        connvert_feature(strategy, rs, self.graph_items[model])
                        label = np.array(json.load(open(runtime_path))['average'])
                        np.savez_compressed(npz_path, x1=var_partition_features, x2=partition_indice, x3=var_num, y=label)
                    else:
                        loaded = np.load(npz_path)
                        var_partition_features, partition_indice, var_num, label = \
                                        loaded['x1'], loaded['x2'], loaded['x3'], loaded['y']

                    if not os.path.isfile(npz_path.replace('.npz', '_pdf.npz')):
                        predefined_features = create_predefined_features(Strategy.deserialize(path=strategy_path), ResourceSpec(resource_file=rs_path), self.predefined_simulators[model])
                        np.savez_compressed(npz_path.replace('.npz', '_pdf.npz'), x4=predefined_features)
                    else:
                        loaded = np.load(npz_path.replace('.npz', '_pdf.npz'))
                        predefined_features = loaded['x4']
                    var_partition_features = np.concatenate([var_partition_features, np.concatenate([predefined_features, np.zeros((MAX_NUM_PARS-predefined_features.shape[0], predefined_features.shape[1]))], 0)], 1)

                    is_aws = int('g3' in strategy_path or 'g4' in strategy_path or 'aws' in strategy_path) # comment here
                    # is_aws = int('vgg16_orca_11_random_rejection-4_trial-100-_expolre-2000_0.83-model_embedding_sim-weight-1_max-par-40/' in strategy_path)
                    # print(model, 'orca' if is_aws == 0 else 'aws', strategy_path.split('/')[-3])
                    features[model][is_aws][0].append(var_partition_features)
                    features[model][is_aws][1].append(partition_indice)
                    features[model][is_aws][2].append(var_num)
                    features[model][is_aws][3].append(label)

        for k, _ in GRAPH_ITEM_PATHS.items():
            for i1 in range(2):
                for i2 in range(4):
                    if len(features[k][i1][i2]) > 1:
                        features[k][i1][i2] = np.stack(features[k][i1][i2]).astype(np.float16)
                        print(k, 'orca' if i1 == 0 else 'aws', features[k][i1][i2].shape)
                    else:
                        features[k][i1][i2] = None

        train_features = np.concatenate([features[model_][is_aws_][0] for model_, is_aws_ in train_patterns if features[model_][is_aws_][0] is not None], 0)
        train_par_indices = np.concatenate([features[model_][is_aws_][1] for model_, is_aws_ in train_patterns if features[model_][is_aws_][1] is not None], 0)
        train_var_nums = np.concatenate([features[model_][is_aws_][2] for model_, is_aws_ in train_patterns if features[model_][is_aws_][2] is not None], 0)
        train_labels = np.concatenate([features[model_][is_aws_][3] for model_, is_aws_ in train_patterns if features[model_][is_aws_][3] is not None], 0)

        if type(valid_patterns[0]) == str and valid_patterns[0] == 'same':
            rng = np.random.RandomState(1)
            permt = rng.permutation(train_features.shape[0])
            split = int(len(permt) * 0.7)
            val_features, val_par_indices, val_var_nums, val_labels = train_features[permt[split:]], train_par_indices[permt[split:]], train_var_nums[permt[split:]], train_labels[permt[split:]]
            train_features, train_par_indices, train_var_nums, train_labels = train_features[permt[:split]], train_par_indices[permt[:split]], train_var_nums[permt[:split]], train_labels[permt[:split]]
        else:
            val_features = np.concatenate([features[model_][is_aws_][0] for model_, is_aws_ in valid_patterns if features[model_][is_aws_][0] is not None], 0)
            val_par_indices = np.concatenate([features[model_][is_aws_][1] for model_, is_aws_ in valid_patterns if features[model_][is_aws_][1] is not None], 0)
            val_var_nums = np.concatenate([features[model_][is_aws_][2] for model_, is_aws_ in valid_patterns if features[model_][is_aws_][2] is not None], 0)
            val_labels = np.concatenate([features[model_][is_aws_][3] for model_, is_aws_ in valid_patterns if features[model_][is_aws_][3] is not None], 0)

            # comment here
            rng = np.random.RandomState(1)
            permt = rng.permutation(val_features.shape[0])
            split = int(len(permt) * 0.7)
            train_features, train_par_indices, train_var_nums, train_labels = np.concatenate([train_features, val_features[permt[:split]]], 0), np.concatenate([train_par_indices, val_par_indices[permt[:split]]], 0), np.concatenate([train_var_nums, val_var_nums[permt[:split]]], 0), np.concatenate([train_labels, val_labels[permt[:split]]], 0)

            val_features, val_par_indices, val_var_nums, val_labels = val_features[permt[split:]], val_par_indices[permt[split:]], val_var_nums[permt[split:]], val_labels[permt[split:]]
        label_max = max(train_labels.max(), val_labels.max())
        label_min = min(train_labels.min(), val_labels.min())
        train_labels = (train_labels-label_min)/(label_max-label_min)
        val_labels = (val_labels-label_min)/(label_max-label_min)
        print(train_features.shape, val_features.shape, train_features.max(), train_features.min(), val_features.max(), val_features.min(), train_labels.max(), val_labels.min()) 

        ## train the model
        trainset = TrainTensorDataset((torch.from_numpy(train_features).half().to(TORCH_DEVICE), torch.from_numpy(train_par_indices).half().to(TORCH_DEVICE), torch.from_numpy(train_var_nums).half().to(TORCH_DEVICE), torch.from_numpy(train_labels).half().to(TORCH_DEVICE)))
        testset = torch.utils.data.TensorDataset(torch.from_numpy(val_features).half().to(TORCH_DEVICE), torch.from_numpy(val_par_indices).half().to(TORCH_DEVICE), torch.from_numpy(val_var_nums).half().to(TORCH_DEVICE), torch.from_numpy(val_labels).half().to(TORCH_DEVICE))
        self.trainloader = torch.utils.data.DataLoader(dataset=trainset, 
                                                  batch_size=BATCH_SIZE, 
                                                  shuffle=True)
        self.testloader = torch.utils.data.DataLoader(dataset=testset, 
                                                  batch_size=32, 
                                                  shuffle=False)

    def train(self, name='', num_epochs=200, checkpoint=None):

        checkpoint_path = 'model_on_{}.ckpt'.format(name)
        print('LSTM layers: ', NUM_RNN_LAYERS, 'score th: ', SCORE_TH, 'lr: ', LR, 'wd: ', WD,'use data aug: ', DATA_AUG, 'OUT_LAYERS: ', OUT_LAYERS, 'IN_LAYERS: ',IN_LAYERS)

        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        model = RankRNN(num_rnn_layers=NUM_RNN_LAYERS, out_layers=OUT_LAYERS, in_layers=IN_LAYERS).to(TORCH_DEVICE)
        if checkpoint:
            model.load_state_dict(torch.load(checkpoint))
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

        best_val_acc = 0.
        for epoch in range(num_epochs):
            if epoch == int(num_epochs*2./5. - 1):
                for param_group in optimizer.param_groups: param_group['lr'] = 3e-4
            if epoch == int(num_epochs*4./5. - 1):
                for param_group in optimizer.param_groups: param_group['lr'] = 1e-4

            labels = []
            outputs = []
            for i, (features_b, par_indices_b, var_nums_b, labels_b) in enumerate(self.trainloader):  
                
                # Forward pass
                outputs_b = model(features_b, par_indices_b, var_nums_b).squeeze()

                par_cnt = (par_indices_b.int() != MAX_NUM_VARS - 1).int().sum(1)
                
                true_comp = (
                    (labels_b[:, None]+SCORE_TH>labels_b[None,:]).int()*(par_cnt[:, None] > par_cnt[None, :]).int() 
                  + (labels_b[:, None]-SCORE_TH>labels_b[None,:]).int()*(par_cnt[:, None] < par_cnt[None, :]).int()
                  + (labels_b[:, None] > labels_b[None,:]).int() * (par_cnt[:, None] == par_cnt[None, :]).int()
                 ) > 0
                true_comp = true_comp.float() * 2 - 1
                pred_comp = outputs_b[:, None] - outputs_b[None, :]
                loss = (1 - true_comp) * pred_comp / 2 + torch.nn.functional.softplus(-pred_comp)
                loss = loss.tril(-1).mean()
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.stem_rnn.parameters(), 0.25)
                optimizer.step()

                outputs.append(outputs_b)
                labels.append(labels_b)

            labels = torch.cat(labels)
            outputs = torch.cat(outputs)
            true_comp = (labels[:, None] > labels[None, :])
            pred_comp = (outputs[:, None] > outputs[None, :])
            equal = (true_comp == pred_comp).int()
            train_acc = equal.tril(-1).sum() * 2. /float(equal.shape[0])/(float(equal.shape[0]) - 1)
            
            with torch.no_grad():
                labels = []
                outputs = []
                for features_b, par_indices_b, var_nums_b, labels_b in self.testloader:
                    
                    # Forward pass
                    outputs_b = model(features_b, par_indices_b, var_nums_b).squeeze()
                    outputs.append(outputs_b)
                    labels.append(labels_b)

                labels = torch.cat(labels)
                outputs = torch.cat(outputs)
                true_comp = (labels[:, None] > labels[None, :])
                pred_comp = (outputs[:, None] > outputs[None, :])
                equal = (true_comp == pred_comp).int()
                acc = equal.tril(-1).sum() * 2. /float(equal.shape[0])/(float(equal.shape[0]) - 1)
                if acc.item() > best_val_acc:
                    best_val_acc = acc.item()
                    if best_val_acc > self.best_acc:
                        print('Saved model @ acc', best_val_acc)
                        torch.save(model.state_dict(), checkpoint_path)
                        self.best_acc = best_val_acc
                    # print('Saved model to {}'.format(checkpoint_path))
                if epoch == num_epochs - 1:
                    print('Epoch: {}, training acc: {:.4f}, test acc: {:.4f}, best acc: {:.4f}, overall best acc: {:.4f}'.format(epoch, train_acc.item(), acc.item(), best_val_acc, self.best_acc))
        return checkpoint_path


if __name__ == '__main__':

    if False:
        trainer = RankNetTrainer()
        trainer.load_data([
            '/users/hzhang2/oceanus_cost_model_training_data/vgg16',
             # '/users/hzhang2/oceanus_cost_model_training_data/ncf-5-11-20', 
             # '/users/hzhang2/oceanus_cost_model_training_data/ncf-5-9-20', 
             # '/users/hzhang2/oceanus_cost_model_training_data/ncf',
             # '/users/hzhang2/oceanus_cost_model_training_data/ncf-5-13-20',
             # '/users/hzhang2/oceanus_cost_model_training_data/bert/bert_large_batch_12_orca_16',
             # '/users/hzhang2/oceanus_cost_model_training_data/bert/bert_large_batch_8_orca_16',
             # '/users/hzhang2/oceanus_cost_model_training_data/bert/bert_large_batch_8_orca_16_group_3',
             # '/users/hzhang2/oceanus_cost_model_training_data/bert/bert_large_batch_8_orca_16_group_4',
             # '/users/hzhang2/oceanus_cost_model_training_data/bert/bert_large_batch_8_orca_16_group_5',
             # '/users/hzhang2/oceanus_cost_model_training_data/bert/bert_large_random_orca_11',
             # '/users/hzhang2/oceanus_cost_model_training_data/bert/bert-aws/bert-large-aws4g4',
             # '/users/hzhang2/oceanus_cost_model_training_data/bert/bert-aws/bert_large_random_search_aws_4_ps_only',
             # '/users/hzhang2/oceanus_cost_model_training_data/densenet', 
             # '/users/hzhang2/oceanus_cost_model_training_data/inceptionv3', 
             # '/users/hzhang2/oceanus_cost_model_training_data/resnet101', 
             # '/users/hzhang2/oceanus_cost_model_training_data/resnet50', 
             ],
            [
              ('vgg16', 1), #('vgg16', 1), 
              # ('ncf', 0), #('ncf', 1), 
              # ('bert_large', 1), #('bert_large', 1), 
              # not used:
              # ('densenet121', 0), ('densenet121', 1), 
              # ('inceptionv3', 0), ('inceptionv3', 1), 
              # ('resnet101', 0), ('resnet101', 1), 
              # ('resnet50', 0), ('resnet50', 1), 
              # ('bert_12l', 0), ('bert_12l', 1), 
              # ('bert_6l', 0), ('bert_6l', 1), 
              # ('bert_3l', 0), ('bert_3l', 1), 
            ], 
            [
              # ('vgg16', 1),
              # ('ncf', 1), 
              # ('bert_large', 1), 
              'same',
            ],
        )
        
        for p2 in [0.01, 0.03]:
            for p3 in [1e-3, 3e-3, 1e-4, 3e-4, 5e-3]:
                for p4 in [1e-3, 1e-4, 3e-4, 5e-4, 5e-5, 2e-3, ]:
                    for p1 in [3, 4, 2]:
                        for p5 in [2, 3]:
                            for p6 in [1, 2]:
                                NUM_RNN_LAYERS, SCORE_TH, LR, WD, IN_LAYERS, OUT_LAYERS = p1, p2, p3, p4, p5, p6
                                checkpoint_path = trainer.train(name='vgg-aws-new-2', num_epochs=200)
        exit()
    else:
        checkpoint_path = '/users/hzhang2/projects/pycharm/zhijie/5-9-2020/oceanus-zhijie/arion/simulator/models/model_on_bert-aws-only.ckpt'
    test_list = [
    '/users/hzhang2/oceanus_cost_model_training_data/bert/bert-aws/bert_large_random_search_aws_4_ps_only',
    # '/users/hzhang2/oceanus_cost_model_training_data/vgg16/vgg16_orca_15',
    # '/users/hzhang2/oceanus_cost_model_training_data/vgg16/vgg_random_orca_11',   #TARGET: 0.9
    # '/users/hzhang2/oceanus_cost_model_training_data/ncf-5-13-20/ncf_large_adam_random_search_aws_4',
    # '/users/hzhang2/oceanus_cost_model_training_data/bert/bert_large_batch_12_orca_16',
    # '/users/hzhang2/oceanus_cost_model_training_data/bert/bert_large_batch_8_orca_16',
    # '/users/hzhang2/oceanus_cost_model_training_data/bert/bert_large_batch_8_orca_16_group_3',
    # '/users/hzhang2/oceanus_cost_model_training_data/bert/bert_large_batch_8_orca_16_group_4',
    # '/users/hzhang2/oceanus_cost_model_training_data/bert/bert_large_batch_8_orca_16_group_5',
    ]
    
    for data_folder in test_list:
        simulator = RankRNNSimulatorPenalty3(GRAPH_ITEM_PATHS[get_model(data_folder)],
                                        4,
                                        2,
                                        1,
                                        batch_size=256,
                                        seq_len=1,
                                        checkpoint=checkpoint_path)

        runtimes_folder = os.path.join(data_folder, 'runtimes')
        results = {}
        averages= []
        scores = []
        strategys = []
        rss = []
        strategy_paths = []
        for name in os.listdir(runtimes_folder):
            strategy_path = os.path.join(data_folder, 'strategies', name)
            rs_path = os.path.join(data_folder, 'resource_specs', name )

            if not os.path.isfile(rs_path):
                rs_path += '.yml' 
            runtime_path = os.path.join(runtimes_folder, name)

            strategy_paths.append(strategy_path)

            with open(runtime_path, 'r') as f:
                runtimes = json.load(f)
            average = np.array(runtimes['average'])

            s = Strategy.deserialize(strategy_path)
            rs = ResourceSpec(resource_file=rs_path)
            strategys.append(s)
            rss.append(rs)

            averages.append(average)

        # for tmp1, tmp2, tmp3 in zip(strategys, rss, strategy_paths):
        #     scores.append(simulator.simulate(tmp1, tmp2, tmp3)[0])
        # print(np.stack(scores).reshape(-1))

        scores = simulator.simulate(strategys, rss)[0]
        print(scores)

        # sorted_by_runtime = {k: v for k, v in sorted(results.items(), key=lambda item: item[1][0])}
        # # sorted_by_scores = {k: v for k, v in sorted(res.items(), key=lambda item: item[1][1])}
        # # sorted_by_latency = {k: v for k, v in sorted(res.items(), key=lambda item: item[1][2])}
        # print('Sorted by runtime.......................')
        # for _, (rt, prediction) in sorted_by_runtime.items():
        #     print('runtime {}  prediction {}'.format(rt, prediction))

        y_train = np.array(averages)
        test_score = np.array(scores)
        true_comp = (y_train.ravel()[:, None] > y_train.ravel()[None, :])
        pred_comp = (test_score.ravel()[:, None] > test_score.ravel()[None, :])
        equal = (true_comp == pred_comp).astype(np.int)
        test_acc = np.tril(equal, -1).sum() * 2. / float(equal.shape[0]) / (float(equal.shape[0]) - 1)

        print('Test {} on {}, acc {:.4f}'.format(checkpoint_path, data_folder.split('/')[-1], test_acc))
