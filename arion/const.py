"""Constants."""

from enum import Enum, auto

import os

DEFAULT_WORKING_DIR = '/tmp/autodist'
DEFAULT_SERIALIZATION_DIR = os.path.join(DEFAULT_WORKING_DIR, 'strategies')
DEFAULT_PORT_RANGE = iter(range(15000, 16000))


class Env(Enum):
    """AutoDist Environment Variables."""

    AUTODIST_WORKER = auto()
    AUTODIST_STRATEGY_ID = auto()


MAX_INT64 = int(2 ** 63 - 1)
COLOCATION_PREFIX = b"loc:@"
AUTODIST_PREFIX = u"AutoDist-Magic-"
AUTODIST_REPLICA_PREFIX = u"%sReplica-" % AUTODIST_PREFIX


class InitOps(Enum):
    """
    List of all AutoDist-defined ops for "initialization".

    Note that the initialization here does not refer to the variable initialization in TensorFlow,
    but the "initialization" before distributed computing for AutoDist.
    """

    MIRROR_VARIABLE_INIT_OP = 'mirror_variable_init_op'


class TraceLevel(Enum):
    """The trace level to be applied in RunnerOption"""

    NO_TRACE = 0
    # SOFTWARE_TRACE = 1
    # HARDWARE_TRACE = 2
    FULL_TRACE = 3
