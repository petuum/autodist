import atexit
import itertools
import os
from multiprocessing import Process

import pytest

from autodist import AutoDist
from autodist.strategy.all_reduce_strategy import AllReduce
from autodist.strategy.parallax_strategy import Parallax
from autodist.strategy.partitioned_ps_strategy import PartitionedPS
from autodist.strategy.ps_lb_strategy import PSLoadBalancing
from autodist.strategy.ps_strategy import PS
from autodist.strategy.partitioned_all_reduce_strategy import PartitionedAR
from autodist.strategy.uneven_partition_ps_strategy import UnevenPartitionedPS
from autodist.strategy.random_axis_partition_all_reduce_strategy import RandomAxisPartitionAR

from .cases import c0, c1, c2, c3, c4, c5, c6, c7, c8

cases = [
    c0,  # TensorFlow basics + placeholder
    c1,  # Keras + iterator
    c2,  # TensorFlow sparse basics + iterator
    c3,  # Keras + placeholder
    c4,  # Control flow while_loop
    c5,  # Keras + numpy input
    c6,  # Dynamic LSTM while loop and other ops
    c7,  # Keras Model.fit & Model.evaluate
    # c8,  # Keras Model building with mixed TF native ops
]
resource_specs = [
    os.path.join(os.path.dirname(__file__), 'resource_specs/r0.yml'),  # single node with 2 GPUs
    os.path.join(os.path.dirname(__file__), 'resource_specs/r2.yml')  # single node with 1 GPU
]
strategies = [
    PS(),
    PartitionedPS(local_proxy_variable=True),
    AllReduce(chunk_size=1),
    PSLoadBalancing(local_proxy_variable=True),
    Parallax(local_proxy_variable=True),
    PartitionedAR(),
    UnevenPartitionedPS(local_proxy_variable=True),
    RandomAxisPartitionAR(chunk_size=4)
]


@pytest.mark.integration
def test_all():
    combinations = itertools.product(resource_specs, strategies)
    for r, s in combinations:
        for c in cases:
            # skip allreduce for sparse variables (TensorFlow bug)
            if type(s) in [AllReduce, PartitionedAR, RandomAxisPartitionAR] and c not in [c0, c1]:
                continue

            def run():
                """This wrapper will handle the AutoDist destructor and garbage collections."""
                try:
                    atexit._clear()  # TensorFlow also uses atexit, but running its exitfuncs cause some issues
                    a = AutoDist(resource_spec_file=r, strategy_builder=s)  # Fixtures in the future
                    c.main(a)
                except Exception:
                    raise
                finally:
                    atexit._run_exitfuncs()

            p = Process(target=run)
            p.start()
            p.join()
            if p.exitcode != 0:
                raise SystemExit(f"FAILED running case {c} with resourcespec {r} and strategy {s}")
