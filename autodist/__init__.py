# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

from tensorflow import version
from tensorflow.python.ops import control_flow_v2_toggles

from .autodist import AutoDist
from .const import ENV
from .patch import PatchTensorFlow
from .utils import logging

logging.set_verbosity(ENV.AUTODIST_MIN_LOG_LEVEL.val)

# Enforce abspath
if sys.argv and os.path.exists(sys.argv[0]) and not os.path.isabs(sys.argv[0]):
    logging.error('AutoDist requires the script path "{}" to be an absolute path to be shared across workers. '
                  'Now exit.'.format(sys.argv[0]))
    sys.exit(1)

# Runtime compatibility checking
COMPAT_TF_VERSIONS = [1.15, 2.2]
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
