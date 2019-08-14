"""User Interface."""

import os
from collections import namedtuple

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.util import tf_contextlib

from autodist.cluster import Cluster
from autodist.const import Env
from autodist.coordinator import Coordinator
from autodist.graph_item import GraphItem
from autodist.resource_spec import ResourceSpec
from autodist.runner import Runner
from autodist.strategy.base import StrategyBuilder

IS_AUTODIST_WORKER = bool(os.environ.get(Env.AUTODIST_WORKER.name))
IS_AUTODIST_CHIEF = not IS_AUTODIST_WORKER


class AutoDist:
    """
    User Interface.

    Notes:
        * Don't initialize session here.
        * Independent of distributed logic.
    """

    CacheKey = namedtuple('CacheKey', ['fn'])

    def __init__(self, resource_spec_file, strategy_name=None):
        self._graph = ops.Graph()
        self._resource_spec = ResourceSpec(resource_file=resource_spec_file)
        self._strategy_name = strategy_name

        self._cluster = None
        self._coordinator = None
        self._cache = {}

    @tf_contextlib.contextmanager
    def scope(self):
        """Scope."""
        with context.graph_mode(), self._graph.as_default():
            yield

    def __del__(self):
        if IS_AUTODIST_CHIEF:
            self._coordinator.join()
            # TODO: active termination instead of passive termination
            # self._cluster.terminate()

    def _build(self, fetches):
        """Core Logic."""
        # this line will traverse the graph and generate necessary stats
        item = GraphItem(self._graph)

        if IS_AUTODIST_CHIEF:
            s = StrategyBuilder.build(item, self._resource_spec, self._strategy_name)
            s.serialize()
        else:
            s = StrategyBuilder.load_strategy()

        self._cluster = Cluster(self._resource_spec)  # which can be also defined with strategy

        runner = Runner(strategy=s, cluster=self._cluster).build(item)

        def run_fn():
            return runner.run(fetches)

        if IS_AUTODIST_CHIEF:
            # we should only have one single coordinator for one single AutoDist() instance scope,
            # even though we could have multiple strategies.
            # TODO: optimize this and serialization
            if not self._coordinator:
                self._coordinator = Coordinator(strategy=s, cluster=self._cluster)

        return run_fn

    def _build_and_run(self, fn, *args, **kwargs):
        # we first assume one fn only build one type of graph
        cache_id = hash(self.CacheKey(fn))
        cached = cache_id in self._cache

        # At the first run of the training function
        if not cached:
            # Build the graph
            fetches = fn(*args, **kwargs)
            # Build the strategy and get the runner with distributed graph
            run_fn = self._build(fetches)
            if IS_AUTODIST_CHIEF:
                self._cluster.start()
                self._coordinator.launch_clients()
            # Cache the runner
            self._cache[cache_id] = run_fn

        run_fn = self._cache[cache_id]
        return run_fn()

    def run(self, fn, *args, **kwargs):
        """
        TFStrategy-like Interface

        Args:
            fn:
            *args:
            **kwargs:

        Returns:

        """
        return self._build_and_run(fn, *args, **kwargs)

    def function(self, fn):
        """
        Decorator Interface.

        # TODO(omkar)
        @AutoDist().function
        def step_fn(*args, **kwargs):
            ...

        step_fn(*args, **kwargs)
        """
        __build_and_run = self._build_and_run

        def wrapper(*args, **kwargs):
            return __build_and_run(fn, *args, **kwargs)

        return wrapper