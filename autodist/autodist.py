"""User Interface."""

import os
import types
from collections import namedtuple

import numpy as np
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.util import tf_contextlib

from autodist.cluster import Cluster
from autodist.const import Env
from autodist.coordinator import Coordinator
from autodist.graph_item import GraphItem
from autodist.kernel.common.utils import get_op_name
from autodist.kernel.graph_transformer import GraphTransformer
from autodist.resource_spec import ResourceSpec
from autodist.runner import Runner, RunnerConfig, WrappedSession
from autodist.strategy.base import StrategyBuilder
from autodist.utils import logging
from autodist.utils.code_transformer import transform

IS_AUTODIST_WORKER = bool(os.environ.get(Env.AUTODIST_WORKER.name))
IS_AUTODIST_CHIEF = not IS_AUTODIST_WORKER


class _AutoDistInterface:

    def __init__(self, resource_spec_file, strategy_name=None, runner_config_file=None):
        self._original_graph = GraphItem(graph=ops.Graph())
        self._resource_spec = ResourceSpec(resource_file=resource_spec_file)
        self._strategy_name = strategy_name
        self._runner_config = RunnerConfig(config_file=runner_config_file)

        self._cluster = None
        self._coordinator = None

    @tf_contextlib.contextmanager
    def scope(self, graph_mode=True):
        """Scope."""
        with self._original_graph.as_default(graph_mode=graph_mode):
            yield

    def build_strategy(self, graph_item):
        """
        Build distributed strategy based on a graph item.

        Args:
            graph_item (GraphItem): AutoDist GraphItem object on which a distributed strategy will be generated.

        Returns:
            (Strategy) Distributed strategy representation object.
        """
        graph_item.info.update(
            variables=graph_item.graph.get_collection(ops.GraphKeys.GLOBAL_VARIABLES),
            table_initializers=graph_item.graph.get_collection(ops.GraphKeys.TABLE_INITIALIZERS)
        )

        if IS_AUTODIST_CHIEF:
            s = StrategyBuilder.build(graph_item, self._resource_spec, self._strategy_name)
            s.serialize()
        else:
            s = StrategyBuilder.load_strategy()
        return s


class _V1(_AutoDistInterface):

    def create_distributed_session(self, *args, **kwargs):
        strategy = self.build_strategy(self._original_graph)

        self._cluster = Cluster(self._resource_spec)  # which can be also defined with strategy

        megatron = GraphTransformer(strategy=strategy, cluster=self._cluster)
        transformed_graph_item = megatron.transform(self._original_graph)

        if IS_AUTODIST_CHIEF:
            # we should only have one single coordinator for one single AutoDist() instance scope,
            # even though we could have multiple strategies.
            # TODO: optimize this and serialization
            if not self._coordinator:
                self._coordinator = Coordinator(strategy=strategy, cluster=self._cluster)
            # if not started
            self._cluster.start()
            self._coordinator.launch_clients()

        return WrappedSession(graph_item=transformed_graph_item, remap_io=megatron.remap_io, cluster=self._cluster)


class _V2Graph(_AutoDistInterface):

    CacheKey = namedtuple('CacheKey', ['fn'])

    def __init__(self, resource_spec_file, strategy_name=None, runner_config_file=None):
        self._cache = {}
        self._args_ph_map = {}
        self._iter_fd = None
        super().__init__(resource_spec_file, strategy_name=strategy_name, runner_config_file=runner_config_file)

    def _build(self, fetches):
        """Core Logic."""
        # this line will traverse the graph and generate necessary stats

        strategy = self.build_strategy(self._original_graph)

        self._cluster = Cluster(self._resource_spec)  # which can be also defined with strategy

        megatron = GraphTransformer(strategy=strategy, cluster=self._cluster)
        transformed_graph_item = megatron.transform(self._original_graph)
        # remap fetches and returns
        new_fetches, _, remap_return_func = megatron.remap_io(transformed_graph_item, fetches)

        runner = Runner(
            graph_item=transformed_graph_item,
            cluster=self._cluster,
            config=self._runner_config
        )

        def run_fn(args, kwargs, args_ph_map, iter_fd):
            try:
                return remap_return_func(runner.run(new_fetches, args, kwargs, args_ph_map, iter_fd))
            except KeyboardInterrupt:
                logging.info('KeyboardInterrupt')
                exit(1)

        if IS_AUTODIST_CHIEF:
            # we should only have one single coordinator for one single AutoDist() instance scope,
            # even though we could have multiple strategies.
            # TODO: optimize this and serialization
            if not self._coordinator:
                self._coordinator = Coordinator(strategy=strategy, cluster=self._cluster)

        return run_fn

    def _get_new_args(self, args, kwargs):
        # Insert placeholders in place of ndarrays
        args_with_ph = []
        kwargs_with_ph = {}
        for i, arg in enumerate(args):
            if isinstance(arg, np.ndarray):
                ph = array_ops.placeholder(dtype=arg.dtype, shape=arg.shape)
                args_with_ph.append(ph)
                self._args_ph_map[get_op_name(ph.name)] = i
            else:
                args_with_ph.append(arg)
        for (k, v) in kwargs.items():
            if isinstance(v, np.ndarray):
                ph = array_ops.placeholder(dtype=v.dtype, shape=v.shape)
                kwargs_with_ph[k] = ph
                self._args_ph_map[get_op_name(ph.name)] = k  # note key name
            else:
                kwargs_with_ph[k] = v
        return tuple(args_with_ph), kwargs_with_ph

    def _build_and_run(self, fn, *args, **kwargs):
        # we first assume one fn only build one type of graph
        cache_id = hash(self.CacheKey(fn))
        cached = cache_id in self._cache

        # At the first run of the training function
        if not cached:
            # Build the graph
            # Feed the args with placeholders
            args_with_ph, kwargs_with_ph = self._get_new_args(args, kwargs)
            fetches = fn(*args_with_ph, **kwargs_with_ph)

            # Build the strategy and get the runner with distributed graph
            run_fn = self._build(fetches)
            # The boundary of graph construction and runtime
            if IS_AUTODIST_CHIEF:
                self._cluster.start()
                self._coordinator.launch_clients()
            # Cache the runner
            self._cache[cache_id] = run_fn

        run_fn = self._cache[cache_id]
        return run_fn(args, kwargs, self._args_ph_map, self._iter_fd)

    def make_dataset_iterator(self, dataset):
        """Takes a dataset or a function and returns an iterator."""
        if isinstance(dataset, types.FunctionType):
            dataset_fn_xform = transform(dataset)
            ds, fd = dataset_fn_xform()
            if fd:
                # we found some tensors that we've replaced with placeholders
                ds_iter = dataset_ops.make_initializable_iterator(ds)
                self._iter_fd = fd
                return ds_iter.get_next()
            else:
                ds_iter = dataset_ops.make_one_shot_iterator(ds)
                return ds_iter.get_next()

        ds_iter = dataset_ops.make_one_shot_iterator(dataset)
        return ds_iter.get_next()

    def function(self, fn):
        """Decorator Interface."""
        __build_and_run = self._build_and_run

        def wrapper(*args, **kwargs):
            return __build_and_run(fn, *args, **kwargs)

        return wrapper


class _V2Eager(_AutoDistInterface):
    """Interface for TensorFlow>=2.x Eager Mode."""

    # def experimental_function(self, fn):
    #     def wrapper(*args, **kwargs):
    #         print(self._original_graph)
    #         return fn(*args, **kwargs)
    #     return wrapper


class AutoDist(_V1, _V2Graph, _V2Eager):
    """User Interface."""
