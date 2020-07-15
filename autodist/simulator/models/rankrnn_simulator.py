"""Strategy RankNetSimulator."""
import glob
import json
import sys
from datetime import datetime
from pathlib import Path
from string import digits

import numpy as np
import os
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import arion
from arion.graph_item import GraphItem
from arion.proto.synchronizers_pb2 import PSSynchronizer, AllReduceSynchronizer
from arion.simulator.models.base import SimulatorBase
from arion.simulator.utils import get_dense_var_bits, get_sparse_var_bits, GIGABITS
from arion.simulator.utils import _resolve_device_address, _max_num_local_replica, _num_local_replica
from arion.strategy.random_sample_strategy import VariableHelper, PartHelper
from arion.strategy.base import Strategy
from arion.resource_spec import ResourceSpec
from arion.cluster import SSHCluster
from arion.kernel.device.resolver import DeviceResolver
from arion.kernel.partitioner import PartitionerConfig
from arion.simulator.models.predefined_simulator import PredefinedSimulator

import torch
import torch.nn as nn

TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# feature settings
MAX_NUM_WORKERS = 16
MAX_NUM_GROUPS = 600
MAX_NUM_VARS = 500
MAX_NUM_PARS = 1500

# model size
FEATURE_SIZE = MAX_NUM_WORKERS+MAX_NUM_GROUPS+15
PARTITION_MLP_HIDDEN = 128
PARTITION_MLP_OUT = 32
STEM_RNN_HIDDEN = 128
BIDIECTIONAL = True
NUM_RNN_LAYERS = 3

# trainer setting
BATCH_SIZE = 64
LR = 3e-4
WD = 3e-4

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
                       bidirectional=BIDIECTIONAL):
        super(RankRNN, self).__init__()
        self.partition_mlp_out = partition_mlp_out
        # self.num_rnn_layers = num_rnn_layers
        self.stem_rnn_hidden = stem_rnn_hidden
        self.partition_mlp = nn.Sequential(nn.Linear(input_size, partition_mlp_hidden),
                                           nn.ReLU(),
                                           # nn.Linear(partition_mlp_hidden, partition_mlp_hidden),
                                           # nn.ReLU(),
                                           nn.Linear(partition_mlp_hidden, partition_mlp_out),
                                           )

        self.stem_rnn = nn.LSTM(partition_mlp_out, stem_rnn_hidden, num_rnn_layers, batch_first=True, bidirectional=bidirectional)
        self.final_fc = nn.Linear(stem_rnn_hidden*num_rnn_layers*(1+int(bidirectional)), 1)

        self.relu = nn.ReLU()
    
    def forward(self, features, par_indices, var_nums):

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
        out = self.final_fc(out)
        return out

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
        # perturbed_device_ids = np.random.permutation(MAX_NUM_WORKERS).astype(np.int32)
        # perturbed_group_ids = np.random.permutation(MAX_NUM_GROUPS).astype(np.int32)
        # mat_device = torch.eye(MAX_NUM_WORKERS, device=x.device, dtype=x.dtype)[perturbed_device_ids]
        # mat_group = torch.eye(MAX_NUM_GROUPS, device=x.device, dtype=x.dtype)[perturbed_group_ids]
        # x = torch.cat([torch.matmul(x[:, :MAX_NUM_WORKERS], mat_device), torch.matmul(x[:, MAX_NUM_WORKERS:MAX_NUM_WORKERS+MAX_NUM_GROUPS], mat_group), x[:, MAX_NUM_WORKERS+MAX_NUM_GROUPS:]], 1)
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

class RankRNNSimulator(SimulatorBase):
    """Simulates strategies for a given graph and resource spec."""

    def __init__(self,
                 original_graph_item_path,
                 fetches=None,
                 batch_size=1,
                 seq_len=1,
                 checkpoint=None):

        super(RankRNNSimulator, self).__init__(original_graph_item_path=original_graph_item_path)
        print("It's using RankNet simulator.")
        self._fetches = fetches
        self._batch_size_per_gpu = batch_size
        self._seq_len = seq_len
        self._checkpoint = checkpoint
        self._predefined_simulator=PredefinedSimulator(original_graph_item_path=original_graph_item_path,
                                                         batch_size=self._batch_size_per_gpu,
                                                         seq_len=self._seq_len)
        if self._checkpoint:
            self._model = RankRNN().to(TORCH_DEVICE)
            self._model.load_state_dict(torch.load(self._checkpoint, map_location=torch.device('cpu')))

    def simulate(self, strategy, resource_spec, strategy_path=None, checkpoint=None):
        cost = self.predict(strategy, resource_spec, strategy_path, checkpoint)
        return cost

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

        return model(var_partition_features, partition_indice, var_num).view(-1).data.cpu().numpy()

class RankNetTrainer():

    def __init__(self,
                 checkpoint=None,
                 batch_size_per_gpu=256,
                 seq_len=1,
                 seed=1):
        self._batch_size_per_gpu = batch_size_per_gpu
        self._seq_len = seq_len
        self.graph_items = {k:GraphItem.deserialize(v) for k, v in GRAPH_ITEM_PATHS.items()}
        self.predefined_simulators = {k: PredefinedSimulator(original_graph_item_path=v,
                                                         batch_size=self._batch_size_per_gpu,
                                                         seq_len=self._seq_len) for k, v in GRAPH_ITEM_PATHS.items()}
        self.model = RankRNN().to(TORCH_DEVICE)
        if checkpoint:
            self.model.load_state_dict(torch.load(checkpoint))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR, weight_decay=WD)
        print("It's using RankNet trainer.")

    def train(self, path_list, train_patterns=[('ncf', 0)], valid_patterns='same', num_epochs=200):

        features = {k: [[[], [], [], []], [[], [], [], []]] for k, _ in GRAPH_ITEM_PATHS.items()}
        for training_path in path_list:
            for path in Path(training_path).rglob('strategies'):
                strategy_paths = glob.glob(os.path.join(path, '*'))
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

                    is_aws = int('g3' in strategy_path or 'g4' in strategy_path or 'aws' in strategy_path or 'vgg_random_orca_11' in strategy_path) # comment here
                    print(model, 'orca' if is_aws == 0 else 'aws', strategy_path.split('/')[-3])
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

        if type(valid_patterns) == str and valid_patterns == 'same':
            permt = np.random.permutation(train_features.shape[0])
            split = int(len(permt) * 0.8)
            val_features, val_par_indices, val_var_nums, val_labels = train_features[permt[split:]], train_par_indices[permt[split:]], train_var_nums[permt[split:]], train_labels[permt[split:]]
            train_features, train_par_indices, train_var_nums, train_labels = train_features[permt[:split]], train_par_indices[permt[:split]], train_var_nums[permt[:split]], train_labels[permt[:split]]
        else:
            val_features = np.concatenate([features[model_][is_aws_][0] for model_, is_aws_ in valid_patterns if features[model_][is_aws_][0] is not None], 0)
            val_par_indices = np.concatenate([features[model_][is_aws_][1] for model_, is_aws_ in valid_patterns if features[model_][is_aws_][1] is not None], 0)
            val_var_nums = np.concatenate([features[model_][is_aws_][2] for model_, is_aws_ in valid_patterns if features[model_][is_aws_][2] is not None], 0)
            val_labels = np.concatenate([features[model_][is_aws_][3] for model_, is_aws_ in valid_patterns if features[model_][is_aws_][3] is not None], 0)

            # comment here
            permt = np.random.permutation(val_features.shape[0])
            split = int(len(permt) * 0.7)
            train_features, train_par_indices, train_var_nums, train_labels = np.concatenate([train_features, val_features[permt[:split]]], 0), np.concatenate([train_par_indices, val_par_indices[permt[:split]]], 0), np.concatenate([train_var_nums, val_var_nums[permt[:split]]], 0), np.concatenate([train_labels, val_labels[permt[:split]]], 0)

            val_features, val_par_indices, val_var_nums, val_labels = val_features[permt[split:]], val_par_indices[permt[split:]], val_var_nums[permt[split:]], val_labels[permt[split:]]

        print(train_features.shape, val_features.shape, train_features.max(), train_features.min(), val_features.max(), val_features.min()) 

        ## train the model
        trainset = TrainTensorDataset((torch.from_numpy(train_features).half().to(TORCH_DEVICE), torch.from_numpy(train_par_indices).half().to(TORCH_DEVICE), torch.from_numpy(train_var_nums).half().to(TORCH_DEVICE), torch.from_numpy(train_labels).half().to(TORCH_DEVICE)))
        testset = torch.utils.data.TensorDataset(torch.from_numpy(val_features).half().to(TORCH_DEVICE), torch.from_numpy(val_par_indices).half().to(TORCH_DEVICE), torch.from_numpy(val_var_nums).half().to(TORCH_DEVICE), torch.from_numpy(val_labels).half().to(TORCH_DEVICE))
        trainloader = torch.utils.data.DataLoader(dataset=trainset, 
                                                  batch_size=BATCH_SIZE, 
                                                  shuffle=True)
        testloader = torch.utils.data.DataLoader(dataset=testset, 
                                                  batch_size=32, 
                                                  shuffle=False)
        best_val_acc = 0.
        checkpoint_path = 'model_train_on_{}-{}_new.ckpt'.format(train_patterns[0][0], 'orca' if train_patterns[0][1] == 0 else 'aws')
        for epoch in range(num_epochs):
            if epoch == int(num_epochs*2./5. - 1):
                for param_group in self.optimizer.param_groups: param_group['lr'] = 3e-4
            if epoch == int(num_epochs*4./5. - 1):
                for param_group in self.optimizer.param_groups: param_group['lr'] = 1e-4

            labels = []
            outputs = []
            for i, (features_b, par_indices_b, var_nums_b, labels_b) in enumerate(trainloader):  
                
                # Forward pass
                outputs_b = self.model(features_b, par_indices_b, var_nums_b).squeeze()
                
                true_comp = (labels_b[:, None] > labels_b[None, :]).float() * 2 - 1
                pred_comp = outputs_b[:, None] - outputs_b[None, :]
                loss = (1 - true_comp) * pred_comp / 2 + torch.nn.functional.softplus(-pred_comp)
                loss = loss.tril(-1).mean()
                
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.stem_rnn.parameters(), 0.25)
                self.optimizer.step()

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
                for features_b, par_indices_b, var_nums_b, labels_b in testloader:
                    
                    # Forward pass
                    outputs_b = self.model(features_b, par_indices_b, var_nums_b).squeeze()
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
                    torch.save(self.model.state_dict(), checkpoint_path)
                    print('Saved model to {}'.format(checkpoint_path))
                print('Epoch: {}, training acc: {:.4f}, test acc: {:.4f}, best acc: {:.4f}'.format(epoch, train_acc.item(), acc.item(), best_val_acc))
        return checkpoint_path


if __name__ == '__main__':
    
    trainer = RankNetTrainer()
    checkpoint_path = trainer.train(
                                    [
                                     # '/users/hzhang2/oceanus_cost_model_training_data/ncf-5-11-20', 
                                     # '/users/hzhang2/oceanus_cost_model_training_data/ncf-5-9-20', 
                                     # '/users/hzhang2/oceanus_cost_model_training_data/ncf',
                                     # '/users/hzhang2/oceanus_cost_model_training_data/ncf-5-13-20',
                                     # '/users/hzhang2/oceanus_cost_model_training_data/densenet', 
                                     # '/users/hzhang2/oceanus_cost_model_training_data/inceptionv3', 
                                     # '/users/hzhang2/oceanus_cost_model_training_data/resnet101', 
                                     # '/users/hzhang2/oceanus_cost_model_training_data/resnet50', 
                                     '/users/hzhang2/oceanus_cost_model_training_data/vgg16',
                                     # '/users/hzhang2/oceanus_cost_model_training_data/bert',
                                     ],
                                    [
                                      # ('ncf', 0), #('ncf', 1), 
                                      # ('densenet121', 0), ('densenet121', 1), 
                                      # ('inceptionv3', 0), ('inceptionv3', 1), 
                                      # ('resnet101', 0), ('resnet101', 1), 
                                      # ('resnet50', 0), ('resnet50', 1), 
                                      # ('bert_12l', 0), ('bert_12l', 1), 
                                      # ('bert_6l', 0), ('bert_6l', 1), 
                                      # ('bert_3l', 0), ('bert_3l', 1), 
                                      # ('bert_large', 0), ('bert_large', 1), 
                                      ('vgg16', 0), #('vgg16', 1), 
                                    ], 
                                    [('vgg16', 1)],
                                    num_epochs=200)
    # checkpoint_path = 'model_train_on_vgg16-orca.ckpt'
    test_list = [
    '/users/hzhang2/oceanus_cost_model_training_data/vgg16/vgg16_orca_15',
    '/users/hzhang2/oceanus_cost_model_training_data/vgg16/vgg_random_orca_11',   #TARGET: 0.9
    # '/users/hzhang2/oceanus_cost_model_training_data/ncf-5-13-20/ncf_large_adam_random_search_aws_4',
    # '/users/hzhang2/oceanus_cost_model_training_data/bert/bert_large_batch_12_orca_16',
    # '/users/hzhang2/oceanus_cost_model_training_data/bert/bert_large_batch_8_orca_16',
    # '/users/hzhang2/oceanus_cost_model_training_data/bert/bert_large_batch_8_orca_16_group_3',
    # '/users/hzhang2/oceanus_cost_model_training_data/bert/bert_large_batch_8_orca_16_group_4',
    # '/users/hzhang2/oceanus_cost_model_training_data/bert/bert_large_batch_8_orca_16_group_5',
    ]
    
    for data_folder in test_list:
        simulator = RankRNNSimulator(GRAPH_ITEM_PATHS[get_model(data_folder)],
                                        batch_size=256,
                                        seq_len=1,
                                        checkpoint=checkpoint_path)

        runtimes_folder = os.path.join(data_folder, 'runtimes')
        results = {}
        averages= []
        scores = []
        for name in os.listdir(runtimes_folder):
            strategy_path = os.path.join(data_folder, 'strategies', name)
            rs_path = os.path.join(data_folder, 'resource_specs', name )
            if not os.path.isfile(rs_path):
                rs_path += '.yml' 
            runtime_path = os.path.join(runtimes_folder, name)

            with open(runtime_path, 'r') as f:
                runtimes = json.load(f)
            average = np.array(runtimes['average'])

            s = Strategy.deserialize(strategy_path)
            rs = ResourceSpec(resource_file=rs_path)
            score = simulator.simulate(s, rs, strategy_path)

            results[name] = (average, score)
            averages.append(average)
            scores.append(score)

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
