import atexit
import itertools
from multiprocessing import Process

import os
import pytest

from autodist import AutoDist
from autodist.strategy.all_reduce_strategy import AllReduce
from autodist.strategy.parallax_strategy import Parallax
from autodist.strategy.partitioned_ps_strategy import PartitionedPS
from autodist.strategy.ps_lb_strategy import PSLoadBalancing
from autodist.strategy.ps_strategy import PS
from .cases import c0, c1, c2, c3, c4, v1_interface

cases = [
    v1_interface,  # v1-style interfaces
    c0,  # TensorFlow 2.0 basics
    c1,  # Keras basics
    c2,  # Sparse basics
    c3,  # Numpy basics
    c4   # Control flow while_loop
]
resource_specs = [
    os.path.join(os.path.dirname(__file__), 'resource_specs/r0.yml'),  # single node with 2 GPUs
    os.path.join(os.path.dirname(__file__), 'resource_specs/r2.yml')  # single node with 1 GPU
]
strategies = [PS(), PartitionedPS(local_proxy_variable=True), AllReduce(),
              PSLoadBalancing(local_proxy_variable=True), Parallax(local_proxy_variable=True)]


@pytest.mark.integration
def test_all():
    combinations = itertools.product(resource_specs, strategies)
    for r, s in combinations:
        for c in cases:
            # skip allreduce for sparse variables (TensorFlow bug)
            if isinstance(s, AllReduce) and c not in [c0, c1]:
                continue
            # skip while_loop case for partitionPS (buggy)
            if isinstance(s, PartitionedPS) and c == c4:
                continue

            def run():
                """This wrapper will handle the AutoDist destructor and garbage collections."""
                atexit._clear()  # TensorFlow also uses atexit, but running its exitfuncs cause some issues
                a = AutoDist(resource_spec_file=r, strategy_builder=s)  # Fixtures in the future
                c.main(a)
                atexit._run_exitfuncs()

            p = Process(target=run)
            p.start()
            p.join()
            assert p.exitcode == 0, f"FAILED running case {c} with resourcespec {r} and strategy {s}"
