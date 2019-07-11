"""Cluster."""

from autodist.const import DEFAULT_PORT_RANGE


class Cluster:
    """Cluster manager for TensorFlow servers."""

    def __init__(self, resource_spec):

        self.cluster_spec = self._get_default_cluster_spec(resource_spec)

    def _get_default_cluster_spec(self, resource_spec):
        print(self.cluster_spec)

        nodes = resource_spec.get_nodes()
        cluster_spec = {
            'worker': [
                '{ip}:{port}'.format(
                    ip=n,
                    port=next(DEFAULT_PORT_RANGE)
                )
            ] for n in nodes
        }
        return cluster_spec

    def start(self):
        """Start."""
        pass

    def terminate(self):
        """Terminate"""
        pass
