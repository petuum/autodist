import atexit
import itertools
from multiprocessing import Process
import pytest

import os

from autodist import AutoDist
from .cases import c0, c1, c2, c3, v1_interface

cases = [
    v1_interface, # v1-style interfaces
    c0,  # TensorFlow 2.0 basics
    c1,  # Keras basics
    c2,  # Sparse basics
    c3   # Numpy basics
]
resource_specs = [
    os.path.join(os.path.dirname(__file__), 'resource_specs/r0.yml'),  # single node with 2 GPUs
    os.path.join(os.path.dirname(__file__), 'resource_specs/r2.yml')  # single node with 1 GPU
    ]
strategies = ['PS', 'PSLoadBalancing', 'PartitionedPS', 'AllReduce', 'Parallax',
              'PSProxy', 'PSLoadBalancingProxy', 'PartitionedPSProxy', 'ParallaxProxy']

@pytest.mark.integration
def test_all():
    combinations = itertools.product(resource_specs, strategies)
    for r, s in combinations:
        for c in cases:
            if s == 'AllReduce' and c not in [c0, c1]:
                continue
            if s in ['AllReduce', 'Parallax', 'ParallaxProxy'] and r == resource_specs[1]:
                continue
            def run():
                """This wrapper will handle the AutoDist destructor and garbage collections."""
                atexit._clear()  # TensorFlow also uses atexit, but running its exitfuncs cause some issues
                a = AutoDist(resource_spec_file=r, strategy_name=s)  # Fixtures in the future
                c.main(a)
                atexit._run_exitfuncs()

            p = Process(target=run)
            p.start()
            p.join()
            assert p.exitcode == 0
