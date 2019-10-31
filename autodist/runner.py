"""Runner."""
import atexit
import os

import yaml
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import timeline, session

import autodist.const
from autodist.utils import logging


# TODO(Hao): could extend this to use tfprof (though I don't
#            see immediate benefit now)
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


        Args:
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

    def __init__(self, graph_item, cluster, config=None):
        self._cluster = cluster
        self._graph_item = graph_item
        self._config = config or RunnerConfig()

        self._session = None
        self._fd = {}
        self._ph_feed_index = {}

    def _clean(self):
        logging.debug('Tearing down clients...')
        # Resetting the variable reference triggers the garbage collection when it jumps out the local
        self._session = None

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
        with self._graph_item.graph.as_default() as graph:
            if not self._session:
                self._create_feed_dict(graph, args_ph_map)

                target = self._cluster.get_local_session_target()
                self._session = session.Session(target=target, config=config_pb2.ConfigProto(
                    allow_soft_placement=True,
                    # log_device_placement=True
                ))
                atexit.register(self._clean)

                # TensorFlow default initializations
                # TODO: Rethink. Should we do this?
                self._session.run(
                    self._graph_item.get_ops_in_graph(self._graph_item.info.initializers)
                )
                self._init_ds_iterator(iter_fd, graph)

            # fill out the feed_dict with new batch
            self._refill_fd(args, kwargs)

            if self._config.trace_level > config_pb2.RunOptions.NO_TRACE:
                options = config_pb2.RunOptions(
                    trace_level=self._config.trace_level
                )
                run_metadata = config_pb2.RunMetadata()
                p = self._session.run(fetches,
                                      options=options,
                                      run_metadata=run_metadata,
                                      feed_dict=self._fd)
                _log_timeline(run_metadata)
            else:
                p = self._session.run(fetches, feed_dict=self._fd)

        return p

    # TODO: v2 way of execution
    # def run_v2():
    # to_func(self.distributed_graph)()


class WrappedSession:
    """Wrapped Session."""

    def __init__(self, graph_item, remap_io, cluster):
        self._graph_item = graph_item
        self._remap_io = remap_io
        self._cluster = cluster

        target = self._cluster.get_local_session_target()
        self._session = session.Session(target=target, graph=self._graph_item.graph, config=config_pb2.ConfigProto(
            allow_soft_placement=True,
            # log_device_placement=True
        ))

        # TensorFlow default initializations
        # TODO: Rethink. Should we do this?
        self._session.run(
            self._graph_item.get_ops_in_graph(self._graph_item.info.initializers)
        )

    def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
        """Wrapped Session.run."""
        new_fetches, new_feed_dict, remap_return_func = self._remap_io(self._graph_item, fetches, feed_dict)
        return remap_return_func(
            self._session.run(
                new_fetches, feed_dict=new_feed_dict, options=options, run_metadata=run_metadata
            )
        )
