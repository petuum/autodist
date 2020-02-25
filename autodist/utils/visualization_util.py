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
