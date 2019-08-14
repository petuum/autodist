import itertools
import os

from autodist import AutoDist
from .cases import c0, c1, c2

cases = [c0, c1]


# pytest integration mark
def test_all():
    resource_specs = [os.path.join(os.path.dirname(__file__), 'resource_specs/r0.yml')]
    strategies = ['PS']
    combinations = itertools.product(resource_specs, strategies)
    for r, s in combinations:
        for c in cases:
            def run():
                """This wrapper will handle the AutoDist destructor and garbage collections."""
                a = AutoDist(resource_spec_file=r, strategy_name=s)  # Fixtures in the future
                c.main(a)

            run()
