"""User Interface."""
import sys
from collections import namedtuple

import os
import types
import numpy as np
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.util import tf_contextlib

from autodist.cluster import Cluster
from autodist.const import Env
from autodist.coordinator import Coordinator
from autodist.graph_item import GraphItem
from autodist.kernel.common.utils import get_op_name
from autodist.kernel.device.resolver import DeviceResolver
from autodist.kernel.graph_transformer import GraphTransformer
from autodist.remapper import Remapper
from autodist.resource_spec import ResourceSpec
from autodist.runner import Runner, RunnerConfig, WrappedSession
from autodist.strategy.base import Strategy, StrategyCompiler
from autodist.utils import logging
from autodist.utils.code_transformer import transform

IS_AUTODIST_WORKER = bool(os.environ.get(Env.AUTODIST_WORKER.name))
IS_AUTODIST_CHIEF = not IS_AUTODIST_WORKER


class _AutoDistInterface:

    def __init__(self, resource_spec_file, strategy_builder, strategy_path=None, runner_config_file=None):
        self._resource_spec = ResourceSpec(resource_file=resource_spec_file)
        self._strategy_builder = strategy_builder
        self._strategy_path = strategy_path
        # TODO: deprecate the runner config
        self._runner_config = RunnerConfig(config_file=runner_config_file)

        self._original_graph_item = None
        # TODO: separate the control
        self._cluster: Cluster = Cluster(self._resource_spec)  # which can be also defined with strategy
        self._coordinator: Coordinator

    @tf_contextlib.contextmanager
    def _scope(self):
        """Forward the context manager of a graph item."""
        with self._original_graph_item.as_default():
            yield

    def build_strategy(self):
        """
        Build distributed strategy based on a graph item.

        Returns:
            (Strategy) Distributed strategy representation object.
        """
        self._original_graph_item.prepare()
        return self._strategy_builder.build(self._original_graph_item, self._resource_spec)

    def _build_or_load_strategy(self):
        if IS_AUTODIST_CHIEF:
            s = self.build_strategy()
            s.serialize()
        else:
            strategy_id = os.environ[Env.AUTODIST_STRATEGY_ID.name]
            s = Strategy.deserialize(strategy_id)
        return s

    def _compile_strategy(self, strategy):
        logging.info('Raw strategy: %s' % strategy)
        device_resolver = DeviceResolver(self._cluster)
        compiled_strategy = StrategyCompiler().set_device_resolver(device_resolver.resolve_to_device_str). \
            compile(strategy)
        logging.info('Compiled strategy: %s' % compiled_strategy)
        return compiled_strategy

    def _setup(self, strategy):
        """Prepare for the execution."""
        if IS_AUTODIST_CHIEF:
            # we should only have one single coordinator for one single AutoDist() instance scope,
            # even though we could have multiple strategies.
            self._coordinator = Coordinator(strategy=strategy, cluster=self._cluster)
            self._cluster.start()
            self._coordinator.launch_clients()


class _GraphModeInterface(_AutoDistInterface):

    def _initialize_graph(self):
        """Postpone the initialization of the member original_graph_item to the scoping time."""
        assert not context.executing_eagerly()
        self._original_graph_item = GraphItem(graph=ops.get_default_graph())


class _V1Graph(_GraphModeInterface):

    def create_distributed_session(self, *args, **kwargs):
        """Create a Session object to execute the default graph in a distributed manner."""
        strategy = self._build_or_load_strategy()
        compiled_strategy = self._compile_strategy(strategy)
        transformed_graph_item = GraphTransformer(
            compiled_strategy=compiled_strategy,
            cluster=self._cluster,
            graph_item=self._original_graph_item
        ).transform()
        remapper = Remapper(compiled_strategy, self._cluster)

        self._setup(strategy)

        # TODO: use the outer args
        return WrappedSession(graph_item=transformed_graph_item, remap_io=remapper.remap_io, cluster=self._cluster)


class _V2Graph(_GraphModeInterface):
    CacheKey = namedtuple('CacheKey', ['fn'])

    def __init__(self, *args, **kwargs):
        self._cache = {}
        self._args_ph_map = {}
        self._iter_fd = None
        super().__init__(*args, **kwargs)

    def _build(self, fetches):
        """Core Logic."""
        strategy = self._build_or_load_strategy()
        compiled_strategy = self._compile_strategy(strategy)
        transformed_graph_item = GraphTransformer(
            compiled_strategy=compiled_strategy,
            cluster=self._cluster,
            graph_item=self._original_graph_item
        ).transform()

        # remap fetches and returns
        remapper = Remapper(compiled_strategy, self._cluster)
        new_fetches, _, remap_return_func = remapper.remap_io(transformed_graph_item, fetches)

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
                sys.exit(1)

        return strategy, run_fn

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
            strategy, run_fn = self._build(fetches)
            # The boundary of graph construction and runtime
            self._setup(strategy)
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
    # TODO: Merge single node eager support from peng-tffunc and peng-eager


class AutoDist(_V1Graph, _V2Graph, _V2Eager):
    """
    AutoDist is a scalable ML engine.

    AutoDist provides user-friendly interfaces to distribute local deep-learning model training
        across multiple processing units with scalability and minimal code changes.
    """

    @tf_contextlib.contextmanager
    def scope(self):
        """
        Returns a context manager capturing the code block to be distributed.

        Returns:
          A context manager.
        """
        if not context.executing_eagerly():
            self._initialize_graph()
        else:
            raise NotImplementedError('AutoDist will support distributed execution under eager mode later.')
        with self._scope():
            yield
