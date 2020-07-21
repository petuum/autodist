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

"""Simulator-related utility functions."""

import glob
import json
import os
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import device_spec
import tensorflow_ranking as tfr

from autodist.utils import logging
from autodist.resource_spec import ResourceSpec
from autodist.strategy.base import Strategy
# from autodist.const import DEFAULT_RUNTIME_SERIALIZATION_DIR, DEFAULT_SERIALIZATION_DIR, \
#     DEFAULT_STRATEGY_JSON_SERIALIZATION_DIR, DEFAULT_RESOURCE_SERIALIZATION_DIR
from autodist.kernel.device.resolver import DeviceResolver


RankingLossKeys = {
	# Names for the ranking based loss functions.
	'pairwise_hinge_loss': tfr.losses.RankingLossKey.PAIRWISE_HINGE_LOSS,
	'pairwise_logistic_loss': tfr.losses.RankingLossKey.PAIRWISE_LOGISTIC_LOSS,
	'pairwise_soft_zero_one_loss': tfr.losses.RankingLossKey.PAIRWISE_SOFT_ZERO_ONE_LOSS,
	'softmax_loss': tfr.losses.RankingLossKey.SOFTMAX_LOSS,
	'sigmoid_cross_entropy_loss': tfr.losses.RankingLossKey.SIGMOID_CROSS_ENTROPY_LOSS,
	'mean_squared_loss': tfr.losses.RankingLossKey.MEAN_SQUARED_LOSS,
	'list_mle_loss': tfr.losses.RankingLossKey.LIST_MLE_LOSS,
	'approx_ndcg_loss': tfr.losses.RankingLossKey.APPROX_NDCG_LOSS,
}

#########
# Online
#########

def laod_from_one_folder(data_folder):
    strategy_folder = '{}/strategies'.format(data_folder)
    strategy_files = glob.glob(os.path.join(strategy_folder, '*'))
    X = []
    Y = []
    for strategy_file in strategy_files:
        # Target
        runtime_file = '/'.join(strategy_file.split('/')[:-2]) + '/runtimes/' + strategy_file.split('/')[-1]
        if not os.path.exists(runtime_file) or not os.path.isfile(runtime_file):
            print('runtime_file does not exist: {}.'.format(runtime_file))
            continue
        runtime = json.load(open(runtime_file, 'r'))
        y = runtime['average']
        resource_file = strategy_file.replace('strategies', 'resource_specs')
        if not os.path.exists(resource_file):
            resource_file += '.yml'
            if not os.path.exists(resource_file):
                resource_file = os.path.join(data_folder, 'resource_spec_files/resource_spec.yml')
                if not os.path.exists(resource_file):
                    continue
        Y.append(y)
        X.append([strategy_file, resource_file])
    print('Data points:{}, data_folder: {}'.format(len(X), data_folder))
    return X, Y


def laod_from_folders(data_dir):
    if isinstance(data_dir, str):
        data_folders = glob.glob("{}/*".format(data_dir), recursive=True)
    elif isinstance(data_dir, list):
        data_folders = data_dir
    else:
        raise ValueError
    print('data_folders', data_folders)
    X = []
    Y = []
    for data_folder in data_folders:
        x, y = laod_from_one_folder(data_folder)
        if len(x) == 0:
            print('strategy_folder does not have files: {}, skipping it.'.format(data_folder))
            continue
        Y.extend(y)
        X.extend(x)
    # Y = np.concatenate(Y, axis=0)
    if len(Y) > 0:
        Y = np.array(Y, dtype=np.float)
        miny = np.min(Y)
        print('min of all Y values: {}'.format(miny))
    else:
        print("no files loaded.")
    return X, Y


##########
# Offline
##########

def laod_from_one_folder_offline(simulation_folder):
    simulation_files = glob.glob(os.path.join(simulation_folder, '*'), recursive=True)
    X = []
    Y = []
    for simulation_file in simulation_files:
        # Features
        try:
            simulation = json.load(open(simulation_file, 'r'))
        except:
            print("Can not read simulation_file: ", simulation_file)
            continue
        x = simulation_file
        # Target
        runtime_file = '/'.join(simulation_file.split('/')[:-2]) + '/runtimes/' + simulation_file.split('/')[-1]
        if not os.path.exists(runtime_file) or not os.path.isfile(runtime_file):
            print('runtime_file does not exist: {}.'.format(runtime_file))
            continue
        runtime = json.load(open(runtime_file, 'r'))
        y = runtime['average']
        Y.append(y)
        X.append(x)
    Y = np.array(Y, dtype=np.float)
    print('Data points:{}, simulation_folder: {}'.format(len(X), simulation_folder))
    return X, Y


def laod_from_folders_offline(data_dir):
    simulation_folders = glob.glob("{}/*/simulations".format(data_dir), recursive=True)
    print('simulation_folders', simulation_folders)
    X = []
    Y = []
    for simulation_folder in simulation_folders:
        x, y = laod_from_one_folder_offline(simulation_folder)
        if len(x) == 0:
            print('simulation folder does not have files: {}, skipping it.'.format(simulation_folder))
            continue
        Y.append(y)
        X.append(x)
    Y = np.concatenate(Y, axis=0)
    miny = np.min(Y)
    print('min of Y values: {}'.format(miny))
    return X, Y


def split_dataset(inputs, shuffle=True, train_ratio=0.7, test_ratio=0.15):
    assert isinstance(inputs, list)
    nb_elements = len(inputs)
    nb_samples = len(inputs[0])
    n_train = int(nb_samples * train_ratio)
    n_test = int(nb_samples * test_ratio)
    shuffled = []
    train = []
    valid = []
    test = []

    if shuffle:
        random_indices = np.random.permutation(list(range(nb_samples)))
        for i in range(nb_elements):
            shuffled_i = [inputs[i][j] for j in random_indices]
            train.append(shuffled_i[:n_train])
            valid.append(shuffled_i[n_train:-n_test])
            test.append(shuffled_i[-n_test:])
    else:
        for i in range(nb_elements):
            train.append(inputs[i][:n_train])
            valid.append(inputs[i][n_train:-n_test])
            test.append(inputs[i][-n_test:])

    return train, valid, test

def read_trial_runs():
    runtime_files = glob.glob(os.path.join(DEFAULT_RUNTIME_SERIALIZATION_DIR, '*'))
    strategy_files = glob.glob(os.path.join(DEFAULT_SERIALIZATION_DIR, '*'))
    strategy_json_files = glob.glob(os.path.join(DEFAULT_STRATEGY_JSON_SERIALIZATION_DIR, '*'))
    resource_files = glob.glob(os.path.join(DEFAULT_RESOURCE_SERIALIZATION_DIR, '*'))
    logging.info(len(runtime_files), len(strategy_files), len(strategy_json_files), len(resource_files))

    trialruns = {}
    for runtime_file in runtime_files:
        strategy_id = runtime_file.split('/')[-1]
        strategy_file = os.path.join(DEFAULT_SERIALIZATION_DIR, strategy_id)
        strategy_json_file = os.path.join(DEFAULT_STRATEGY_JSON_SERIALIZATION_DIR, strategy_id)
        resource_file = os.path.join(DEFAULT_RESOURCE_SERIALIZATION_DIR, strategy_id)
        if not os.path.exists(strategy_file):
            logging.info("strategy_file not found, skip it: {}".format(strategy_file))
            continue
        if not os.path.exists(strategy_json_file):
            logging.info("strategy_json_file not found, skip it: {}".format(strategy_json_file))
            continue
        if not os.path.exists(resource_file):
            logging.info("resource_file not found, skip it: {}".format(resource_file))
            continue

        trialruns[strategy_id] = {
            'runtime': json.load(open(runtime_file, 'r')),
            'strategy': Strategy.deserialize(strategy_id),
            'strategy_json': json.load(open(strategy_json_file, 'r')),
            'resource_spec': ResourceSpec(resource_file=resource_file),
        }

    logging.info("Total number of trials: {}".format(len(trialruns)))
    return trialruns


DTYPE2BITS = {
    tf.float16: 16,
    "tf.float16": 16,
    "<dtype: 'float16'>": 16,
    tf.float32: 32,
    'tf.float32': 32,
    "<dtype: 'float32'>": 32,
    "<dtype: 'float32_ref'>": 32,
    tf.float64: 64,
    'tf.float64': 64,
    "<dtype: 'float64'>": 64,
    tf.bfloat16: 16,
    'tf.bfloat16': 16,
    "<dtype: 'bfloat16'>": 16,
    tf.complex64: 64,
    'tf.complex64': 64,
    "<dtype: 'complex64'>": 64,
    tf.complex128: 128,
    'tf.complex128': 128,
    "<dtype: 'complex128'>": 128,
    tf.int8: 8,
    'tf.int8': 8,
    "<dtype: 'int8'>": 8,
    tf.uint8: 8,
    'tf.uint8': 8,
    "<dtype: 'uint8'>": 8,
    tf.uint16: 16,
    'tf.uint16': 16,
    "<dtype: 'uint16'>": 16,
    tf.uint32: 32,
    'tf.uint32': 32,
    "<dtype: 'uint32'>": 32,
    tf.uint64: 64,
    'tf.uint64': 64,
    "<dtype: 'uint64'>": 64,
    tf.int16: 16,
    'tf.int16': 16,
    "<dtype: 'int16'>": 16,
    tf.int32: 32,
    'tf.int32': 32,
    "<dtype: 'int32'>": 32,
    tf.int64: 64,
    'tf.int64': 64,
    "<dtype: 'int64'>": 64,
    tf.bool: 1,
    'tf.bool': 1,
    "<dtype: 'bool'>": 1,
    tf.string: 1,  # todo: confirm
    'tf.string': 1,  # todo: confirm
    "<dtype: 'string'>": 1,  # todo: confirm
    tf.qint8: 8,
    'tf.qint8': 8,
    "<dtype: 'qint8'>": 8,
    tf.quint8: 8,
    'tf.quint8': 8,
    "<dtype: 'quint8'>": 8,
    tf.qint16: 16,
    'tf.qint16': 16,
    "<dtype: 'qint16'>": 16,
    tf.quint16: 16,
    'tf.quint16': 16,
    "<dtype: 'quint16'>": 16,
    tf.qint32: 32,
    'tf.qint32': 32,
    "<dtype: 'qint32'>": 32,
    tf.resource: 0,  # its tensor shape is either [] or [None] todo: confirm
    'tf.resource': 0,  # its tensor shape is either [] or [None] todo: confirm
    "<dtype: 'resource'>": 0,  # its tensor shape is either [] or [None] todo: confirm
}

GIGABITS = np.float(1e+9)
INFINITY = 1e+9
NUM_RUNS = 500
GPU_TO_CPU_BANDWIDTH = 1000 # Gbps


def pad_list(l, max_len):
    return l + [0.0] * (max_len - len(l))


def get_dtype_bits(dtype):
    return DTYPE2BITS[dtype] if dtype in DTYPE2BITS else DTYPE2BITS[str(dtype)]


def get_dense_var_bits(size, dtype):
    return size * get_dtype_bits(dtype)


def get_sparse_var_bits(size):
    # same size of values, indices, dense_shape
    return size * (get_dtype_bits(tf.float32) + 2 * get_dtype_bits(tf.int64)) \
           + 2 * get_dtype_bits(tf.int64)


def _resolved_devices_on_diff_machine(device1, device2):
    # e.g., '/job:worker/task:1/device:CPU:0', '/job:worker/task:1/GPU:0'
    node1 = ':'.join(device1.split('/')[:-1])
    node2 = ':'.join(device2.split('/')[:-1])
    return node1 != node2


# def _resolve_device_address(device: str, device_resolver: DeviceResolver):
#     # change real ip address to /job:worker/task:0
#     if not device:
#         return device
#     parts = device.split(':')
#     if parts and parts[0] in device_resolver._address_to_tasks:
#         resolved_device = device_resolver._address_to_tasks[parts[0]][0]
#         resolved = '/job:{}/task:{}/device:'.format(resolved_device['job'], resolved_device['task'])
#         resolved = resolved + ':'.join(parts[-2:])
#         return resolved
#     else:
#         raise ValueError("cannot resolve device: {} using device_resolver: {}".format(
#             device, device_resolver._address_to_tasks))


# def _num_local_replica(host, replicas, cluster):
#     # host: e.g., '/job:worker/task:0/device:CPU:0'
#     replica_devices = {device_spec.DeviceSpecV2.from_string(r) for r in replicas}
#     host_device = device_spec.DeviceSpecV2.from_string(host)
#     num_local_replica = sum(1 for d in replica_devices
#                             if cluster.get_address_from_task(d.job, d.task) ==
#                             cluster.get_address_from_task(host_device.job, host_device.task))
#     return num_local_replica
#
#
# def _max_num_local_replica(replicas, cluster):
#     replica_devices = {device_spec.DeviceSpecV2.from_string(r) for r in replicas}
#     replica_hosts = {cluster.get_address_from_task(d.job, d.task) for d in replica_devices}
#     max_num_local_replica = 0
#     for host in replica_hosts:
#         num_local_replica = sum(1 for d in replica_devices
#                                 if cluster.get_address_from_task(d.job, d.task) == host)
#         max_num_local_replica = max(max_num_local_replica, num_local_replica)
#     return max_num_local_replica


def _strip_var_name(name):
    # strip prefix
    if not name:
        return name
    name = name.split('/')
    if 'Replica' in name[0]:  # remove prefix
        name = name[1:]
    if name and 'part' in name[-1]:  # remove '/part_1' if using partitioned ps
        name = name[:-1]
    name = '/'.join(name)
    name = name.split(':')[0]  # remove ':0'.
    return name
