"""Runner."""
from datetime import datetime

from tensorflow.core.protobuf import config_pb2
from tensorflow.python import ops
from tensorflow.python.client.session import Session
from tensorflow.python.framework import dtypes
from tensorflow.python.ops.variables import global_variables_initializer, local_variables_initializer, Variable
from tensorflow.python.ops.lookup_ops import tables_initializer
from tensorflow.python.summary.writer import writer
from tensorflow.python.training.saver import import_meta_graph

from autodist.kernel.common import utils
from autodist.kernel.device.resolver import DeviceResolver
from autodist.kernel.replication.replicator import Replicator
from autodist.kernel.synchronization.synchronizer import Synchronizer
from autodist.strategy.base import StrategyCompiler


class Runner:
    """Runner in worker process."""

    def __init__(self, strategy, cluster):
        self._strategy = strategy
        self.c = cluster
        self.transformed_graph = ops.Graph()
        self._is_built = False
        self.session = None

    def build(self, item):
        """
        Build distributed graph.

        Args:
            item (GraphItem): wrapper of TensorFlow graph.
        """
        def log_graph(name, graph):
            graph_name = datetime.now().strftime("%Y%m%d-%H%M%S")
            writer.FileWriter('./logs/{}'.format(graph_name + name), graph=graph)

        log_graph('original', graph=item.graph)
        # open('./graphdefs/{}'.format(graph_name+'original'), 'w+').write(str(item._graph.as_graph_def()))

        # Compile Strategy
        print('# Raw strategy:', self._strategy)
        device_resolver = DeviceResolver(self.c)
        strategy = StrategyCompiler().set_device_resolver(device_resolver.resolve_to_device_str).compile(self._strategy)
        # strategy = self._strategy
        print('# Compiled strategy:', strategy)

        # Create Synchronizers for each node in the strategy
        synchronizers = {
            name: Synchronizer.create(node['synchronizer']['type'], **node['synchronizer']['config'])
            for name, node in strategy.node_config.items()
        }

        # Replicate the graph (both in-graph and between-graph)
        r = Replicator(
            config=strategy.graph_config.get('replicas'),
            cluster=self.c,
            synchronizers=synchronizers
        )

        final_item = r.apply(item)

        self._finialize_build(final_item)
        log_graph('transformed', graph=self.transformed_graph)

        return self

    def _finialize_build(self, graph_item):
        with self.transformed_graph.as_default():
            import_meta_graph(graph_item.meta_graph)

    def run(self, fetches, feed=None):
        """Execute distributed graph."""
        with self.transformed_graph.as_default() as graph:
            if not self.session:
                target = self.c.get_local_session_target()
                self.session = Session(target=target, config=config_pb2.ConfigProto(
                    allow_soft_placement=True,
                    # log_device_placement=True
                ))
                self.session.run(global_variables_initializer())
                self.session.run(local_variables_initializer())
                if ops.get_collection(ops.GraphKeys.TABLE_INITIALIZERS):
                    self.session.run(tables_initializer())

            new_fetches = []
            for f in fetches:
                if isinstance(f, ops.Tensor):
                    new_fetch = graph.get_tensor_by_name(f.name)
                elif isinstance(f, ops.Operation):
                    new_fetch = graph.get_operation_by_name(f.name)
                elif isinstance(f, Variable):
                    handle = graph.get_tensor_by_name(f.name)
                    if handle.dtype is dtypes.resource:
                        # Resource Var
                        new_fetch = utils.get_resource_read_variable_tensor(handle)
                    else:
                        # Ref Var
                        new_fetch = handle
                else:
                    raise TypeError('Fetch type {} not supported.'.format(type(f)))
                assert graph.is_fetchable(new_fetch)
                new_fetches.append(new_fetch)

            p = self.session.run(new_fetches)

            # TODO: if people wants to run it eagerly
            # to_func(self.distributed_graph)()
        return p
