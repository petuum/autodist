"""Runner."""
import os
from datetime import datetime
import yaml

from tensorflow.core.protobuf import config_pb2
from tensorflow.python import ops
from tensorflow.python.client.session import Session
from tensorflow.python.framework import dtypes
from tensorflow.python.ops.variables import global_variables_initializer, local_variables_initializer, Variable
from tensorflow.python.ops.lookup_ops import tables_initializer
from tensorflow.python.summary.writer import writer
from tensorflow.python.client import timeline

import autodist.const
from autodist.kernel.common import resource_variable
from autodist.kernel.device.resolver import DeviceResolver
from autodist.kernel.replication.replicator import Replicator
from autodist.kernel.synchronization.synchronizer import Synchronizer
from autodist.strategy.base import StrategyCompiler
from autodist.utils import logging  # pylint: disable=useless-import-alias

logging.set_verbosity('DEBUG')


# TODO(Hao): could extend this to use tfprof (though I don't
# see immediate benefit now)
def _log_timeline(run_metadata):
    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    # TODO(Hao): add a runner step count and use it here.
    with open(os.path.join(autodist.const.DEFAULT_WORKING_DIR, "traces_timeline.json"), "w") as f:
        f.write(chrome_trace)


# Future: convert this to protobuf
class RunnerConfig():
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
        self.trace_level = autodist.const.TraceLevel.NO_TRACE
        self.log_graph = True
        if config_file and os.path.isfile(config_file):
            self._from_config_file(config_file)

    def _from_config_file(self, config_file):
        config = yaml.safe_load(open(config_file, 'r'))
        if 'trace_level' in config:
            self.trace_level = autodist.const.TraceLevel(config.pop('trace_level'))
        self.log_graph = config.pop('log_graph', False)


class Runner:
    """Runner in worker process."""

    def __init__(self, strategy, cluster, config=None):
        self._strategy = strategy
        self._cluster = cluster
        self._is_built = False
        self.original_graph_item = None
        self.transformed_graph_item = None
        self.session = None
        self._fd = {}
        self._ph_feed_index = {}
        self.config = config or RunnerConfig()

    def build(self, item):
        """
        Build distributed graph.

        Args:
            item (GraphItem): wrapper of TensorFlow graph.
        """
        self.original_graph_item = item

        def log_graph(name, graph):
            graph_name = datetime.now().strftime("%Y%m%d-%H%M%S")
            graph_default_path = os.path.join(autodist.const.DEFAULT_WORKING_DIR,
                                              'logs/{}'.format(graph_name + name))
            writer.FileWriter(graph_default_path, graph=graph)

        if self.config.log_graph:
            log_graph('original', graph=item.graph)
        # open('./graphdefs/{}'.format(graph_name+'original'), 'w+').write(str(item._graph.as_graph_def()))

        # Compile Strategy
        logging.info('Raw strategy: %s' % self._strategy)
        device_resolver = DeviceResolver(self._cluster)
        strategy = StrategyCompiler().set_device_resolver(device_resolver.resolve_to_device_str).compile(self._strategy)
        # strategy = self._strategy
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
        if self.config.log_graph:
            log_graph('transformed', graph=self.transformed_graph_item.graph)

        return self

    def _finalize_build(self, graph_item):
        self.transformed_graph_item = graph_item
        self._is_built = self.transformed_graph_item is not None

    def _run_by_name(self, name, session=None):
        """Run graph by op or tensor name."""
        session = self.session if not session else session
        graph = session.graph
        try:
            op = graph.get_operation_by_name(name)
            logging.info('######\nRun by name:\n{}######'.format(op))
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

    def run(self, fetches, args=None, kwargs=None, args_ph_map=None):
        """Execute distributed graph."""
        assert self._is_built
        with self.transformed_graph_item.graph.as_default() as graph:
            if not self.session:
                target = self._cluster.get_local_session_target()
                self.session = Session(target=target, config=config_pb2.ConfigProto(
                    allow_soft_placement=True,
                    # log_device_placement=True
                ))
                # TensorFlow default initializations
                # TODO: Rethink. Should we do this?
                self.session.run(global_variables_initializer())
                self.session.run(local_variables_initializer())
                self._create_feed_dict(graph, args_ph_map)
                if ops.get_collection(ops.GraphKeys.TABLE_INITIALIZERS):
                    self.session.run(tables_initializer())

                # AutoDist initializations
                for op in autodist.const.InitOps:
                    self._run_by_name(op.value)

            new_fetches = []
            for f in fetches:
                if isinstance(f, ops.Tensor):
                    new_fetch = graph.get_tensor_by_name(f.name)
                    # # TODO: make this a reasonable fetch for replicated tensors
                    # new_fetch = {
                    #     'Replica-0': ops.prepend_name_scope(f.name, replica_prefix(0)),
                    #     'Replica-1': ops.prepend_name_scope(f.name, replica_prefix(1)),
                    # }
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
                # assert graph.is_fetchable(new_fetch)
                new_fetches.append(new_fetch)

            # fill out the feed_dict with new batch
            self._refill_fd(args, kwargs)

            if self.config.trace_level is autodist.const.TraceLevel.FULL_TRACE:
                options = config_pb2.RunOptions(
                    trace_level=config_pb2.RunOptions.FULL_TRACE
                )
                run_metadata = config_pb2.RunMetadata()
                p = self.session.run(new_fetches,
                                     options=options,
                                     run_metadata=run_metadata,
                                     feed_dict=self._fd)
                _log_timeline(run_metadata)
            else:
                p = self.session.run(new_fetches, feed_dict=self._fd)

            # TODO: if people wants to run it eagerly
            # to_func(self.distributed_graph)()
        return p
