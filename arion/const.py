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


class ENV(Enum):
    """
    AutoDist Environment Variables.

    Note: If there's a cleaner/better way to do this, please do it.

    This is an Enum because in some instances we need to access the `name`
    field of a property. AFAIK, if we were to do this with normal variables
    this would require some sort of `inspect`ion.

    Since we use each environment variable in such different ways,
    we just make the enum value a lambda that will be called by
    our own `val` property.

    For example, we want `AUTODIST_IS_TESTING` to be a bool (`True` or `False`)
    depending on the string set as an environment variable, so the lambda returns
    a comparison of the string and `"True"`.
    """

    AUTODIST_WORKER = auto(), lambda v: v or ""                          # noqa: E731
    AUTODIST_STRATEGY_ID = auto(), lambda v: v or ""                     # noqa: E731
    AUTODIST_MIN_LOG_LEVEL = auto(), lambda v: v or "INFO"               # noqa: E731
    AUTODIST_IS_TESTING = auto(), lambda v: (v or "False") == "True"     # noqa: E731
    AUTODIST_DEBUG_REMOTE = auto(), lambda v: (v or "False") == "True"   # noqa: E731
    AUTODIST_PATCH_TF = auto(), lambda v: (v or "False") == "True"       # noqa: E731
    SYS_DATA_PATH = auto(), lambda v: v or ""                         # noqa: E731
    SYS_RESOURCE_PATH = auto(), lambda v: v or ""                     # noqa: E731

    @property
    def val(self):
        """Return the output of the lambda on the system's value in the environment."""
        # pylint: disable=invalid-envvar-value, unpacking-non-sequence
        _, default_fn = self.value
        return default_fn(os.getenv(self.name))


MAX_INT64 = int(2 ** 63 - 1)
MAX_INT32 = int(2 ** 31 - 1)
COLOCATION_PREFIX = b"loc:@"
AUTODIST_PREFIX = u"AutoDist-"
AUTODIST_REPLICA_PREFIX = u"%sReplica-" % AUTODIST_PREFIX
AUTODIST_TO_DELETE_SCOPE = u"to-delete"
