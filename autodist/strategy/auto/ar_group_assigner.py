from collections import OrderedDict

import numpy as np


def chunk_group_assigner(ar_shards, chunk_size=1):
    assignments = {}
    for i, shard_name in enumerate(ar_shards):
        assignments[shard_name] = i // chunk_size
    assert(len(ar_shards)) == len(assignments)
    return assignments


def christy_group_assigner(ar_shards, var_helpers, num_group):
    """A probabilistic assigner that  tries to put each ring with balanced message size"""
    assignments = {}

    sorted_ar_shards = OrderedDict(sorted(ar_shards.items(), key=lambda x: var_helpers[x[0]].byte_size, reverse=True))
    cur_loads = [0.0 for i in range(num_group)]
    for shard_name in sorted_ar_shards:
        total_loads = sum(cur_loads)
        balanced_loads = [total_loads / num_group for _ in range(num_group)]
        space = np.array([balanced_load - cur_load for balanced_load, cur_load in zip(balanced_loads, cur_loads)])

        e_x = np.exp(space-np.max(space))
        accept_prob = e_x / e_x.sum()

        des = np.random.choice(range(0, num_group), 1, p=accept_prob)[0]
        assignments[shard_name] = des
        cur_loads[des] += var_helpers[shard_name].byte_size
    assert(len(ar_shards)) == len(assignments)
    # entropy = calcuate_entropy(cur_loads)
    # best_entropy = calcuate_entropy(balanced_loads)
    # print('entropy {} vs. max entropy {}'.format(entropy, best_entropy))
    return assignments

def ordered_balanced_group_assigner(ar_shards, var_helpers, num_group):
    """Greedy assigner that create balanced loads following a given var order."""
    assignments = {}

    # get total size
    total_loads = 0.0
    for shard_name in ar_shards:
        total_loads += var_helpers[shard_name].byte_size

    avg_load = total_loads / num_group

    cur_bucket = 0
    loads = [0 for _ in range(num_group)]
    for shard_name in ar_shards:
        if loads[cur_bucket] >= avg_load:
            cur_bucket += 1
        if loads[cur_bucket] < avg_load:
            assignments[shard_name] = cur_bucket
            loads[cur_bucket] += var_helpers[shard_name].byte_size
    assert(len(ar_shards) == len(assignments))
    return assignments