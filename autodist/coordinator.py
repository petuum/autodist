"""Coordinator."""

from autodist.cluster import Cluster


class Coordinator:
    """Coordinator to manager TF cluster and processes of workers."""

    def __init__(self, strategy, resource_spec):
        self.cluster = Cluster(resource_spec)

        self._extract_from(strategy)

    def launch(self):
        """Launch."""
        # TF Clusters start
        self.cluster.start()  # start the tf cluster

        # Subprocesses for clients start
        # self.clients.rmote -> "python train.py"

    def terminate(self):
        """Terminate."""
        pass

    def _extract_from(self, strategy):

        # how_many_clients <- strategy.property
        pass
