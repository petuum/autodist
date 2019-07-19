"""User Interface."""

import os

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_contextlib

from autodist.const import Env
from autodist.coordinator import Coordinator
from autodist.item import Item
from autodist.resource_spec import ResourceSpec
from autodist.runner import Runner
from autodist.strategy.base import StrategyBuilder

IS_AUTODIST_WORKER = os.environ.get(Env.AUTODIST_WORKER.name) == 'true'


class AutoDist:
    """
    User Interface.

    Notes:
        * Don't initialize session here.
        * Independent of distributed logic.
    """

    def __init__(self, resource_spec_file, strategy_name=None):
        self._graph = ops.Graph()
        self._resource_spec = ResourceSpec(resource_file=resource_spec_file)
        self._strategy_name = strategy_name

    @tf_contextlib.contextmanager
    def scope(self):
        """Scope."""
        with context.graph_mode(), self._graph.as_default():
            yield

    def _run(self, fetches):
        """Core Logic."""
        # this line will traverse the graph and generate necessary stats
        item = Item(self._graph)

        if not IS_AUTODIST_WORKER:
            s = StrategyBuilder.build(item, self._resource_spec, self._strategy_name)
        else:
            s = StrategyBuilder.load_strategy()

        logging.info(s)

        if not IS_AUTODIST_WORKER:
            print('# Master')
            Coordinator(
                strategy=s,
                resource_spec=self._resource_spec
            ).launch_cluster().launch_clients()
        else:
            print('# Worker')
            Runner(s, self._resource_spec).build(item).run(fetches)

    # def run(self, fetches, feed_dict):
    #     """
    #     AutoDist().run(fetches, feed_dict)
    #     """
    #     return self._run(fetches, feed_dict=fetches)

    def run(self, step_fn, *args, **kwargs):
        """
        Debugging Interface. Will be deprecated soon.

        Args:
            step_fn ([type]): [description]

        Returns:
            [type]: [description]
        """
        fetches = step_fn(*args, **kwargs)
        return self._run(fetches)

    def function(self, fn):
        """
        Decorate train fn.

        TODO(peng.wu)
        @AutoDist().function
        def step_fn(*args, **kwargs):
            ...

        step_fn(*args, **kwargs)
        """
        def wrapper(*args, **kwargs):
            # get the single-node graph
            # (?) tf.funtion helps you identify the fetches
            # self._run(fetches)
            fetches = fn(*args, **kwargs)
            print('Inner Function:', fetches)
            self._run(fetches)
            return fetches

        return wrapper
