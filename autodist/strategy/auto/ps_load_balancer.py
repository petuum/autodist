from collections import OrderedDict

import numpy as np


def calcuate_entropy(loads):
    distribution = loads / np.sum(loads)
    distribution = distribution + 1e-4
    entropy = - np.sum(distribution * np.log2(distribution))
    return entropy

def greedy_load_balancer(ps_shards, resource_spec, var_helpers, sort_by_size=False):
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

