import itertools
import pytest

import os

from autodist import AutoDist
from .cases import c0


def test_dist():
    resource_file = os.path.join(os.path.dirname(__file__), 'resource_specs/r1.yml')
    strategy = 'PS'
    a = AutoDist(resource_spec_file=resource_file, strategy_name=strategy)
    c0.main(a)
