"""Runner."""

from autodist.cluster import Cluster


class Runner:
    """Runner in worker process."""

    def __init__(self, strategy, resource_spec):
        self.c = Cluster(resource_spec)

    def build(self, graph):
        """Build distributed graph."""
        print(self.c)
        # self.distributed_graph = kernerls(graph)
        # devices setter
        # paritioner
        # synchronizer + replicator(?)
        print('Potatos!!!')
        return self

    def run(self, fetches, feed=None):
        """Execute distributed graph."""
        # Session(self.distributed_graph).run(fetchs)
        # to_func(self.distributed_graph)()
        pass
