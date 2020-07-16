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

"""PS load balancers."""

from collections import OrderedDict

import numpy as np


def calcuate_entropy(loads):
    distribution = loads / np.sum(loads)
    distribution = distribution + 1e-4
    entropy = - np.sum(distribution * np.log2(distribution))
    return entropy


def greedy_load_balancer(ps_shards, resource_spec, var_helpers, sort_by_size=False):
    """
    A greedy load balancer that places the next largest load on the least loaded server.
    Args:
        ps_shards:
        resource_spec:
        var_helpers:
        sort_by_size:

    Returns:

    """
    # no randomness
    assignments = {}
    reduction_device_names = [k for k, _ in resource_spec.cpu_devices]
    loads = {ps: 0.0 for ps in reduction_device_names}

    sorted_ps_shards = ps_shards
    if sort_by_size:
        sorted_ps_shards = OrderedDict(sorted(ps_shards.items(),
                                              key=lambda x: var_helpers[x[0]].byte_size, reverse=True))

    for shard_name in sorted_ps_shards:
        sorted_ps = sorted(loads, key=loads.get)
        destination = sorted_ps[0]
        assignments[shard_name] = destination
        loads[destination] += var_helpers[shard_name].byte_size
    return assignments


def christy_load_balancer(ps_shards, resource_spec, var_helpers, sort_by_size=False):
    """
    A randomized greedy load balancer. It places the variable by sampling from a multinomial distribution
    correlated with their current load status -- node with least loads will have highest probability being
    sampled.

    Args:
        ps_shards:
        resource_spec:
        var_helpers:
        sort_by_size:

    Returns:

    """
    # Sample destination based on a distributed calculated based on loads and available bandwidth
    reduction_device_names = [k for k, _ in resource_spec.cpu_devices]
    loads = {ps: 0.0 for ps in reduction_device_names}
    assignments = {}

    loads = sorted(list(loads.items()), key=lambda x: x[0])
    ps = [load[0] for load in loads]
    bandwidth = [resource_spec.network_bandwidth[p.split(':')[0]] for p in ps]
    total_bandwidth = sum(bandwidth)
    cur_loads = [float(load[1]) for load in loads]

    sorted_ps_shards = ps_shards
    if sort_by_size:
        sorted_ps_shards = OrderedDict(sorted(ps_shards.items(),
                                              key=lambda x: var_helpers[x[0]].byte_size, reverse=True))

    for shard_name in sorted_ps_shards:
        total_load = sum(cur_loads)  # + var_load
        balanced_loads = [total_load * b / total_bandwidth for b in bandwidth]
        space = np.array([balanced_load - cur_load for balanced_load, cur_load in zip(balanced_loads, cur_loads)])

        # softmax
        e_x = np.exp(space - np.max(space))
        accept_prob = e_x / e_x.sum()

        # sample according to current load
        des = np.random.choice(ps, 1, p=accept_prob)[0]
        assignments[shard_name] = des

        cur_loads[ps.index(des)] += var_helpers[shard_name].byte_size
    assert (len(ps_shards) == len(assignments))

    # entropy = calcuate_entropy(cur_loads)
    # best_entropy = calcuate_entropy(balanced_loads)
    # print('entropy {} vs. max entropy {}'.format(entropy, best_entropy))
    return assignments
