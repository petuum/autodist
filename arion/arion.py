"""User Interface."""
import atexit
from collections import namedtuple

import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.util import tf_contextlib

from autodist.network import Cluster, SSHCluster
from autodist.const import ENV
from autodist.coordinator import Coordinator
from autodist.graph_item import GraphItem
from autodist.kernel.device.resolver import DeviceResolver
from autodist.kernel.graph_transformer import GraphTransformer
from autodist.remapper import Remapper
from autodist.resource_spec import ResourceSpec
from autodist.runner import WrappedSession
from autodist.strategy.base import Strategy, StrategyCompiler
from autodist.utils import logging

IS_AUTODIST_WORKER = bool(ENV.AUTODIST_WORKER.val)
IS_AUTODIST_CHIEF = not IS_AUTODIST_WORKER


class _AutoDistInterface:

    def __init__(self, resource_spec_file, strategy_builder, strategy_path=None):
        self._resource_spec = ResourceSpec(resource_file=resource_spec_file)
        self._strategy_builder = strategy_builder
        self._strategy_path = strategy_path

        self._original_graph_item = None
        # TODO: separate the control
        self._cluster: Cluster = SSHCluster(self._resource_spec)  # which can be also defined with strategy
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
            strategy_id = ENV.AUTODIST_STRATEGY_ID.val
            assert strategy_id
            s = Strategy.deserialize(strategy_id)
        return s

    def _compile_strategy(self, strategy):
        logging.debug('Raw strategy: %s' % strategy)
        device_resolver = DeviceResolver(self._cluster)
        compiled_strategy = StrategyCompiler().set_device_resolver(device_resolver.resolve_to_device_str). \
            compile(strategy)
        logging.debug('Compiled strategy: %s' % compiled_strategy)
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
        graph_transformer = GraphTransformer(
            compiled_strategy=compiled_strategy,
            cluster=self._cluster,
            graph_item=self._original_graph_item
        )
        transformed_graph_item = graph_transformer.transform()
        remapper = Remapper(graph_transformer, transformed_graph_item)

        # End: Graph Construction, Begin: Running
        self._setup(strategy)
        return transformed_graph_item, remapper

    def _create_distributed_session(self):
        """Create a Session object to execute the default graph in a distributed manner."""
        transformed_graph_item, remapper = self._build()

        _distributed_session = WrappedSession(
            cluster=self._cluster,
            graph_item=transformed_graph_item,
            remapper=remapper,
        )

        def _del(sess=_distributed_session):
            """Enforce the sess to be closed before the cluster termination in the atexit stack."""
            sess.__del__()
            logging.info('Closing session...')
        atexit.register(_del)

        return _distributed_session


class _V1Graph(_GraphModeInterface):

    def create_distributed_session(self):
        """Create a Session object to execute the default graph in a distributed manner."""
        return self._create_distributed_session()


class _V2Graph(_GraphModeInterface):
    CacheKey = namedtuple('CacheKey', ['fn'])

    def __init__(self, *args, **kwargs):
        self._cache = {}
        self._ph_feed_index = {}
        super().__init__(*args, **kwargs)

    def _get_new_args(self, args, kwargs):
        # TODO: currently this follows Keras convention to treat the first dimension as batch dim
        #   However, we should use the tf.function to handle the complete cases of input signatures
        _warning_msg = 'AutoDist treats the first dimension of autodist.function input with shape {} as batch dimenstion'

        # Insert placeholders in place of ndarrays
        args_with_ph = []
        kwargs_with_ph = {}
        for i, arg in enumerate(args):
            if isinstance(arg, np.ndarray):
                logging.warning(_warning_msg.format(arg.shape))
                ph = array_ops.placeholder(dtype=arg.dtype, shape=(None, *arg.shape[1:]))
                args_with_ph.append(ph)
                self._ph_feed_index[ph] = i
            else:
                args_with_ph.append(arg)
        for (k, v) in kwargs.items():
            if isinstance(v, np.ndarray):
                logging.warning(_warning_msg.format(v.shape))
                ph = array_ops.placeholder(dtype=v.dtype, shape=(None, *v.shape[1:]))
                kwargs_with_ph[k] = ph
                self._ph_feed_index[ph] = k
            else:
                kwargs_with_ph[k] = v
        return tuple(args_with_ph), kwargs_with_ph

    def _refill_fd(self, *args, **kwargs):
        """
        Refill the FeedDict with the numeric fn args and kwargs.

        Use the index populated in _ph_feed_index to quickly assign the right
          argument to the right placeholder.
        """
        fd = {}
        for ph, index in self._ph_feed_index.items():
            if isinstance(index, int):
                fd[ph] = args[index]
            else:
                fd[ph] = kwargs[index]
        return fd

    def _build_fn(self, fn, *args, **kwargs):
        # Build the graph
        # Feed the args with placeholders
        args_with_ph, kwargs_with_ph = self._get_new_args(args, kwargs)
        refill_feed_dict = self._refill_fd
        fetches = fn(*args_with_ph, **kwargs_with_ph)

        # Build the strategy and get the runner with distributed graph
        session = self._create_distributed_session()

        def run_fn(*args, **kwargs):
            # fill out the feed_dict with new batch
            feed_dict = refill_feed_dict(*args, **kwargs)
            return session.run(fetches, feed_dict)

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
                if _cache:
                    raise NotImplementedError("AutoDist currently only stably supports "
                                              "one 'autodist.function' across the scope.")
                # Cache the runner
                _cache[cache_id] = _build_fn(fn, *args, **kwargs)
                atexit.register(lambda: _cache.pop(cache_id))
            return _cache[cache_id](*args, **kwargs)

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
