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


MAX_INT64 = int(2 ** 63 - 1)
COLOCATION_PREFIX = 'loc:@'
BINARY_ENCODED_COLOCATION_PREFIX = b"loc:@"
MIRROR_VARIABLE_INIT_OP = "auto_parallel_replicated_var_init_op"
UPDATE_OP_VAR_POS = 0
SPARSE_AVERAGE_BY_COUNTER = 1
SPARSE_NO_AVERAGE = 3
