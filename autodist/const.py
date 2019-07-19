"""Constants."""

import os
from enum import Enum, auto

DEFAULT_WORKING_DIR = '/tmp/autodist'
DEFAULT_SERIALIZATION_DIR = os.path.join(DEFAULT_WORKING_DIR, 'strategies')
DEFAULT_PORT_RANGE = iter(range(15000, 16000))


class Env(Enum):
    """AutoDist Environment Variables."""

    AUTODIST_WORKER = auto()
    AUTODIST_STRATEGY_ID = auto()
