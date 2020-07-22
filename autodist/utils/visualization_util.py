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

"""Utility functions for graph visualization."""
import os

from tensorflow.python.summary.writer import writer

import autodist.const
from autodist.utils import logging


def log_graph(graph, name):
    """
    Log the graph on Tensorboard.

    Args:
        graph: the tensorflow graph to be plotted on Tensorboard
        name: the name of the graph
    """
    directory = os.path.join(autodist.const.DEFAULT_WORKING_DIR, "graphs")
    os.makedirs(directory, exist_ok=True)
    p = os.path.join(directory, name)
    writer.FileWriter(p, graph=graph)
    logging.debug('Graph summary written to: %s' % p)
