"""User Interface."""
import atexit
import sys
from collections import namedtuple

import os
import numpy as np
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
from autodist.resource_spec import ResourceSpec
from autodist.runner import WrappedSession
from autodist.strategy.base import Strategy, StrategyCompiler
from autodist.utils import logging

IS_AUTODIST_WORKER = bool(os.environ.get(Env.AUTODIST_WORKER.name))
IS_AUTODIST_CHIEF = not IS_AUTODIST_WORKER


class _AutoDistInterface:

    def __init__(self, resource_spec_file, strategy_builder, strategy_path=None):
        self._resource_spec = ResourceSpec(resource_file=resource_spec_file)
        self._strategy_builder = strategy_builder
        self._strategy_path = strategy_path

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

    def _build(self):
        strategy = self._build_or_load_strategy()
        compiled_strategy = self._compile_strategy(strategy)
        transformed_graph_item = GraphTransformer(
            compiled_strategy=compiled_strategy,
            cluster=self._cluster,
            graph_item=self._original_graph_item
        ).transform()

        # End: Graph Construction, Begin: Running
        self._setup(strategy)

        return compiled_strategy, transformed_graph_item

    def _create_distributed_session(self):
        """Create a Session object to execute the default graph in a distributed manner."""
        compiled_strategy, transformed_graph_item = self._build()
        return WrappedSession(
            cluster=self._cluster,
            graph_item=transformed_graph_item,
            compiled_strategy=compiled_strategy,
        )


class _V1Graph(_GraphModeInterface):

    def create_distributed_session(self):
        """Create a Session object to execute the default graph in a distributed manner."""
        return self._create_distributed_session()


class _V2Graph(_GraphModeInterface):
    CacheKey = namedtuple('CacheKey', ['fn'])

    def __init__(self, *args, **kwargs):
        self._cache = {}
        self._args_ph_map = {}
        self._fd = {}
        self._ph_feed_index = {}
        super().__init__(*args, **kwargs)

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

    def _create_feed_dict(self, graph, args_ph_map):
        """
        Create the FeedDict.

        We have to remap the inputs (args and kwargs) to the right placeholder
          created in the *replicated* graph. args_ph_map holds a map of placeholder
          *names* to the argument tensor. Note that there are N copies of a
          placeholder for N replicas and we have to feed all of them with tensors.
          The mapping looks like original graph -> replicated graph -> argument
          index.
        """
        for op in graph.get_operations():
            if op.type == "Placeholder":
                ph = op.outputs[0]
                ph_name = op.name.split('/')[-1]
                if ph_name in args_ph_map:
                    self._fd[ph] = None
                    self._ph_feed_index[ph] = args_ph_map[ph_name]

    def _refill_fd(self, args, kwargs):
        """
        Refill the FeedDict with the numeric fn args and kwargs.

        Use the index populated in _ph_feed_index to quickly assign the right
          argument to the right placeholder.
        """
        for x in self._fd:
            if isinstance(self._ph_feed_index[x], int):
                self._fd[x] = args[self._ph_feed_index[x]]
            else:
                self._fd[x] = kwargs[self._ph_feed_index[x]]
        return self._fd

    def _build_fn(self, fn, *args, **kwargs):
        # Build the graph
        # Feed the args with placeholders
        args_with_ph, kwargs_with_ph = self._get_new_args(args, kwargs)
        fetches = fn(*args_with_ph, **kwargs_with_ph)

        # Build the strategy and get the runner with distributed graph
        session = self._create_distributed_session()
        self._create_feed_dict(session._graph_item.graph, self._args_ph_map)
        refill_feed_dict = self._refill_fd

        def run_fn(args, kwargs):
            try:
                # fill out the feed_dict with new batch
                feed_dict = refill_feed_dict(args, kwargs)
                return session.run(fetches, feed_dict)
            except KeyboardInterrupt:
                logging.info('KeyboardInterrupt')
                sys.exit(1)
        return run_fn

    def function(self, fn):
        """Decorator Interface."""
        _cache = self._cache
        _build_fn = self._build_fn

        def wrapper(*args, **kwargs):
            # we first assume one fn only build one type of graph
            cache_id = hash(_V2Graph.CacheKey(fn))
            cached = cache_id in _cache

            # At the first run of the training function
            if not cached:
                # Cache the runner
                _cache[cache_id] = _build_fn(fn, *args, **kwargs)
                atexit.register(lambda: _cache.pop(cache_id))
            return _cache[cache_id](args, kwargs)

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
