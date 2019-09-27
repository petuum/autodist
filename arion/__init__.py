import os

from autodist.patch import PatchTensorFlow
from .utils import logging
from .autodist import AutoDist

if os.environ.get('AUTODIST_PATCH_TF', '') == '1':
    PatchTensorFlow.patch_var_reading()
PatchTensorFlow.init_gradient_handler()

logging.set_verbosity('DEBUG')
