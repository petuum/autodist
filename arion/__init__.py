import sys

from tensorflow import version
from tensorflow.python.ops import control_flow_v2_toggles

from .autodist import AutoDist
from .const import ENV
from .patch import PatchTensorFlow
from .utils import logging

logging.set_verbosity(ENV.AUTODIST_MIN_LOG_LEVEL.val)

# Runtime compatibility checking
COMPAT_TF_VERSIONS = [1.15, 2.1]
float_major_minor_tf_version = float(version.VERSION[:version.VERSION.rfind('.')])
if not COMPAT_TF_VERSIONS[0] <= float_major_minor_tf_version <= COMPAT_TF_VERSIONS[1]:
    logging.error('AutoDist is only compatible with `tensorflow-gpu>={}, <={}`, but the current version is {}'.format(
        COMPAT_TF_VERSIONS[0], COMPAT_TF_VERSIONS[1],
        float_major_minor_tf_version
    ))
    sys.exit(1)
logging.debug('AutoDist is now running on TensorFlow {}'.format(version.VERSION))

# Disable tensorflow control flow version 2 (which AutoDist does not support as of now).
# Use control flow version 1 instead.
control_flow_v2_toggles.disable_control_flow_v2()
logging.warning('AutoDist has disabled TensorFlow control_flow_v2 in favor of control_flow_v1')

PatchTensorFlow.patch_optimizers()
