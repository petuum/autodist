import atexit
import itertools
from multiprocessing import Process
import pytest

import os

from autodist import AutoDist
from .cases import c0, c1, c2, c3

cases = [
    c0,  # TensorFlow 2.0 basics
    c1,  # Keras basics
    c2,  # Sparse basics
    c3   # Numpy basics
]
resource_specs = [
    os.path.join(os.path.dirname(__file__), 'resource_specs/r0.yml'),
    # os.path.join(os.path.dirname(__file__), 'resource_specs/r1.yml'),
    ]
strategies = ['PS', 'PSLoadBalancing', 'PartitionedPS']

@pytest.mark.integration
def test_all():
    combinations = itertools.product(resource_specs, strategies)
    for r, s in combinations:
        for c in cases:
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
