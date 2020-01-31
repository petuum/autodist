import os
import sys

from tensorflow import version
from tensorflow.python.ops import control_flow_v2_toggles

from .autodist import AutoDist
from .const import Env
from .patch import PatchTensorFlow
from .utils import logging

logging.set_verbosity(os.environ.get(Env.AUTODIST_MIN_LOG_LEVEL.name, 'DEBUG'))

# Runtime compatibility checking
COMPAT_VERSIONS = [1.15, 2.0]
float_major_minor_version = float(version.VERSION[:version.VERSION.rfind('.')])
if float_major_minor_version not in COMPAT_VERSIONS:
    logging.error('AutoDist is only compatible with tensorflow-gpu=={}, but the current version is {}'.format(
        COMPAT_VERSIONS,
        float_major_minor_version
    ))
    sys.exit(1)
logging.info('AutoDist is now on TensorFlow {}'.format(version.VERSION))

# Disable tensorflow control flow version 2 (which AutoDist does not support as of now).
# Use control flow version 1 instead.
control_flow_v2_toggles.disable_control_flow_v2()
logging.warning('AutoDist has disabled TensorFlow control_flow_v2 for control_flow_v1')

if os.environ.get('AUTODIST_PATCH_TF', '') == '1':
    PatchTensorFlow.patch_var_reading()
PatchTensorFlow.patch_optimizers()
