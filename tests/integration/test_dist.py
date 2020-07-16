import itertools
import subprocess

import os

from .single_run import STRATEGIES_FOR_DISTRIBUTED_TESTS

cases = [
    "c0",  # TensorFlow 2.0 basics
    # "c1",  # Keras basics
    # "c2",  # Sparse basics
    # "c3",  # Numpy basics
    # "c4",  # Control flow while_loop
    # "c9",  # Staleness
]

resource_specs = [
    os.path.join(os.path.dirname(__file__), 'resource_specs/r1.yml'),
    # os.path.join(os.path.dirname(__file__), 'resource_specs/r3.yml'),
    # os.path.join(os.path.dirname(__file__), 'resource_specs/r4.yml')
]


def test_dist():
    combinations = itertools.product(resource_specs, STRATEGIES_FOR_DISTRIBUTED_TESTS.keys(), cases)
    for r, s, c in combinations:
        # skip allreduce for sparse variables (TensorFlow bug)
        if s in ['AllReduce', 'PartitionedAR', 'AllReduce_2', 'RandomAxisPartitionAR'] and c not in ["c0", "c1"]:
            continue
        # skip while_loop case for partitionPS (buggy)
        if s in ['PartitionedPS', 'UnevenPartitionedPS'] and c == 'c4':
            continue
        # only run c9 for staleness
        if (c == "c9" and 'stale' not in s) or (c != "c9" and 'stale' in s):
            continue
        p = os.path.join(os.path.dirname(__file__), 'single_run.py')
        cmd = ("python {} --case={} --strategy={} --resource={}").format(p, c, s, r)
        print("=====> test starts!")
        print("=====> cmd is {}".format(cmd))
        status = subprocess.run(args=cmd, shell=True)
        assert status.returncode == 0, "{}, {}, {}".format(r, s, c)
        print("=====> test success")
