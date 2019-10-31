"""Constants."""

from enum import Enum, auto

import os

DEFAULT_WORKING_DIR = '/tmp/autodist'
os.makedirs(DEFAULT_WORKING_DIR, exist_ok=True)
DEFAULT_SERIALIZATION_DIR = os.path.join(DEFAULT_WORKING_DIR, 'strategies')
os.makedirs(DEFAULT_SERIALIZATION_DIR, exist_ok=True)
DEFAULT_PORT_RANGE = iter(range(15000, 16000))

# For Allreduce and Collective Ops
DEFAULT_GROUP_LEADER = '/job:worker/replica:0/task:0'


class Env(Enum):
    """AutoDist Environment Variables."""

    AUTODIST_WORKER = auto()
    AUTODIST_STRATEGY_ID = auto()


MAX_INT64 = int(2 ** 63 - 1)
MAX_INT32 = int(2 ** 31 - 1)
COLOCATION_PREFIX = b"loc:@"
AUTODIST_PREFIX = u"AutoDist-"
AUTODIST_REPLICA_PREFIX = u"%sReplica-" % AUTODIST_PREFIX
AUTODIST_TO_DELETE_SCOPE = u"to-delete"


class InitOps(Enum):
    """
    List of all AutoDist-defined ops for "initialization".

    Note that the initialization here does not refer to the variable initialization in TensorFlow,
    but the "initialization" before distributed computing for AutoDist.
    """

    MIRROR_VARIABLE_INIT_OP = 'mirror_variable_init_op'
