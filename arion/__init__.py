import os

from .kernel.experimental.patch import PatchTensorFlow
from .utils import logging
from .autodist import AutoDist

if os.environ.get('AUTODIST_PATCH_TF', '') == '1':
    PatchTensorFlow.patch_var_reading()

logging.set_verbosity('DEBUG')
