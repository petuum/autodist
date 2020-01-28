import os

import tensorflow as tf

from autodist.patch import PatchTensorFlow
from .const import Env
from .utils import logging
from .autodist import AutoDist

logging.set_verbosity(os.environ.get(Env.AUTODIST_MIN_LOG_LEVEL.name, 'DEBUG'))

# Disable tensorflow control flow version 2 (which AutoDist does not support as of now).
# Use control flow version 1 instead.
tf.compat.v1.disable_control_flow_v2()

if os.environ.get('AUTODIST_PATCH_TF', '') == '1':
    PatchTensorFlow.patch_var_reading()
PatchTensorFlow.patch_optimizers()
