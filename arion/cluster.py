"""Cluster."""

from autodist.const import DEFAULT_PORT_RANGE


class Cluster:
    """Cluster manager for TensorFlow servers."""

    def __init__(self, resource_spec):

        self.cluster_spec = self._get_default_cluster_spec(resource_spec)

    @staticmethod
    def _get_default_cluster_spec(resource_spec):

        return {
            'worker': [
                '{ip}:{port}'.format(
                    ip=n,
                    port=next(DEFAULT_PORT_RANGE)
                )
            ] for n in resource_spec.nodes
        }

    def start(self):
        """Start."""
        pass

    def terminate(self):
        """Terminate"""
        pass
