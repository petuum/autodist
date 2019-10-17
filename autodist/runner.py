"""Runner."""
import atexit
import os

import yaml
from tensorflow.core.protobuf import config_pb2
from tensorflow.python import ops
from tensorflow.python.client import timeline
from tensorflow.python.client.session import Session
from tensorflow.python.framework import dtypes
from tensorflow.python.ops.variables import Variable

import autodist.const
from autodist.graph_item import GraphItem
from autodist.kernel.common import resource_variable
from autodist.kernel.common.utils import replica_prefix
from autodist.kernel.device.resolver import DeviceResolver
from autodist.kernel.replication.replicator import Replicator
from autodist.kernel.synchronization.synchronizer import Synchronizer
from autodist.strategy.base import StrategyCompiler
from autodist.utils import logging
from autodist.utils import visualization_util


# TODO(Hao): could extend this to use tfprof (though I don't
# see immediate benefit now)
def _log_timeline(run_metadata, name='timeline', step=0):
    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    directory = os.path.join(autodist.const.DEFAULT_WORKING_DIR, "traces")
    os.makedirs(directory, exist_ok=True)
    # TODO(Hao): add a runner step count and use it here.
    p = os.path.join(directory, "{}_{}.json".format(name, step))
    with open(p, "w") as f:
        f.write(chrome_trace)
        logging.info('Traced timeline written to: %s' % p)


# Future: convert this to protobuf
class RunnerConfig:
    """Meta configurations of the runner."""

    def __init__(self, config_file=None):
        """
        Read runner configurations from a config file.

        Be default, no trace will be captured and the log level is set to INFO.
        It is fine to not provide a config file on non-chief nodes; in that case, those nodes will
        adopt the default RunnerConfig.


        Args:AutoDist
            config_file (string, optional): file path to the config file . Defaults to None.
        """
        self.trace_level = config_pb2.RunOptions.NO_TRACE
        self.log_graph = True
        if config_file and os.path.isfile(config_file):
            self._from_config_file(config_file)

    def _from_config_file(self, config_file):
        config = yaml.safe_load(open(config_file, 'r'))
        self.trace_level = getattr(config_pb2.RunOptions.TraceLevel,
                                   config.pop('trace_level', 'NO_TRACE'))
        self.log_graph = config.pop('log_graph', False)


class Runner:
    """Runner in worker process."""

    def __init__(self, strategy, cluster, config=None):
        self._strategy = strategy
        self._cluster = cluster
        self._is_built = False
        self._transformed_graph_item = None
        self._config = config or RunnerConfig()

        self._session = None
        self._fd = {}
        self._ph_feed_index = {}
        self._fetches = []

    def _clean(self):
        logging.debug('Tearing down clients...')
        # Resetting the variable reference triggers the garbage collection when it jumps out the local
        self._session = None

    def build(self, item: GraphItem):
        """
        Build distributed graph.

        Args:
            item (GraphItem): wrapper of TensorFlow graph.
        """
        assert not self._is_built

        if self._config.log_graph:
            visualization_util.log_graph(graph=item.graph, name='original')

        # Compile Strategy
        logging.info('Raw strategy: %s' % self._strategy)
        device_resolver = DeviceResolver(self._cluster)
        strategy = StrategyCompiler().set_device_resolver(device_resolver.resolve_to_device_str).compile(self._strategy)
        logging.info('Compiled strategy: %s' % strategy)

        # Create Synchronizers for each node in the strategy
        synchronizers = {
            name: Synchronizer.create(node['synchronizer']['type'], **node['synchronizer']['config'])
            for name, node in strategy.node_config.items()
        }

        # Replicate the graph (both in-graph and between-graph)
        r = Replicator(
            config=strategy.graph_config.get('replicas'),
            cluster=self._cluster,
            synchronizers=synchronizers
        )

        final_item = r.apply(item)

        self._finalize_build(final_item)

        if self._config.log_graph:
            visualization_util.log_graph(graph=self._transformed_graph_item.graph, name='transformed')

        return self

    def _finalize_build(self, graph_item):
        self._transformed_graph_item = graph_item
        self._is_built = self._transformed_graph_item is not None
        logging.info('Successfully built transformed graph')

    def _run_by_name(self, name, session=None):
        """Run graph by op or tensor name."""
        session = self._session if not session else session
        graph = session.graph
        try:
            op = graph.get_operation_by_name(name)
            logging.info('Run by name:\n{}'.format(op))
            session.run(op)
        except KeyError:
            pass

    # We have to remap the inputs (args and kwargs) to the right placeholder
    # created in the *replicated* graph. args_ph_map holds a map of placeholder
    # *names* to the argument tensor. Note that there are N copies of a
    # placeholder for N replicas and we have to feed all of them with tensors.
    # The mapping looks like original graph -> replicated graph -> argument
    # index
    def _create_feed_dict(self, graph, args_ph_map):
        for op in graph.get_operations():
            if op.type == "Placeholder":
                ph = op.outputs[0]
                ph_name = op.name.split('/')[-1]
                if ph_name in args_ph_map:
                    self._fd[ph] = None
                    self._ph_feed_index[ph] = args_ph_map[ph_name]

    # use the index populated in _ph_feed_index to quickly assign the right
    # argument to the right placeholder
    def _refill_fd(self, args, kwargs):
        for x in self._fd:
            if isinstance(self._ph_feed_index[x], int):
                self._fd[x] = args[self._ph_feed_index[x]]
            else:
                self._fd[x] = kwargs[self._ph_feed_index[x]]

    def _remap_fetches(self, graph, fetches):
        for f in fetches:
            if isinstance(f, ops.Tensor):
                try:
                    new_fetch = graph.get_tensor_by_name(f.name)
                except KeyError:
                    # TODO: make this a reasonable fetch for replicated tensors
                    replica_f_name = ops.prepend_name_scope(f.name, replica_prefix(0))
                    logging.warning('Fetching replicated tensor "{}" now gets: "{}"'.format(f.name, replica_f_name))
                    new_fetch = graph.get_tensor_by_name(replica_f_name)
            elif isinstance(f, ops.Operation):
                new_fetch = graph.get_operation_by_name(f.name)
            elif isinstance(f, Variable):
                handle = graph.get_tensor_by_name(f.name)
                if handle.dtype is dtypes.resource:
                    # Resource Var
                    new_fetch = resource_variable.get_read_var_tensor(handle.op)
                else:
                    # Ref Var
                    new_fetch = handle
            else:
                raise TypeError('Fetch type {} not supported.'.format(type(f)))
            self._fetches.append(new_fetch)

    def _init_ds_iterator(self, iter_fd, graph):
        if not iter_fd:
            return

        # we create new fd for the replicated graph
        def remap(old_fd):
            fd = {}
            for op in graph.get_operations():
                if op.type == "Placeholder":
                    for k, v in old_fd.items():
                        if op.name.split('/')[-1] == k.name.split(':')[0]:
                            fd[op.outputs[0]] = v
            return fd

        remap_fd = remap(iter_fd)
        # initialize the replicated iterators with the new fd
        for op in graph.get_operations():
            if op.type == "MakeIterator":
                self._session.run(op, feed_dict=remap_fd)

    def run(self, fetches, args=None, kwargs=None, args_ph_map=None, iter_fd=None):
        """Execute distributed graph."""
        assert self._is_built
        with self._transformed_graph_item.graph.as_default() as graph:
            if not self._session:
                self._create_feed_dict(graph, args_ph_map)
                self._remap_fetches(graph, fetches)

                target = self._cluster.get_local_session_target()
                self._session = Session(target=target, config=config_pb2.ConfigProto(
                    allow_soft_placement=True,
                    # log_device_placement=True
                ))
                atexit.register(self._clean)

                # TensorFlow default initializations
                # TODO: Rethink. Should we do this?
                self._session.run(
                    self._transformed_graph_item.get_ops_in_graph(self._transformed_graph_item.info.initializers)
                )
                self._init_ds_iterator(iter_fd, graph)
                # AutoDist initializations
                for op in autodist.const.InitOps:
                    self._run_by_name(op.value)

            # fill out the feed_dict with new batch
            self._refill_fd(args, kwargs)

            if self._config.trace_level > config_pb2.RunOptions.NO_TRACE:
                options = config_pb2.RunOptions(
                    trace_level=self._config.trace_level
                )
                run_metadata = config_pb2.RunMetadata()
                p = self._session.run(self._fetches,
                                      options=options,
                                      run_metadata=run_metadata,
                                      feed_dict=self._fd)
                _log_timeline(run_metadata)
            else:
                p = self._session.run(self._fetches, feed_dict=self._fd)
        return p

    # TODO: v2 way of execution
    # def run_v2():
    # to_func(self.distributed_graph)()
