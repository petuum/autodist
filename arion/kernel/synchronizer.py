"""Synchronizer."""


class Config:
    """Default Config."""

    pass


class Synchronizer:
    """Synchronizer."""

    # a static context to record the load balancing status
    # and make some adjustment when necessary
    context = {}

    def __init__(self):
        self.config = {}

    def apply(self, var, config, graph, resource_spec):
        """Abstract."""
        print(self.config)


# TODO(trevin)
class PSSynchronizer(Synchronizer):
    """PS Synchronizer."""

    def in_graph_apply(self):
        """Apply the synchronizer given in-graph replication."""
        print(self.config)

    def between_graph_apply(self, var, config, graph, resource_spec):
        """
        Apply the synchronizer given between-graph replication.

        Modify `graph`, add synchronizer for `var` in `graph` following
        `config`, and return a modified graph

        Args:
            var ([type]): [description]
            config ([type]): [description]
            graph ([type]): [description]
            resource_spec ([type]): [description]

        Returns:
            [type]: [description]
        """
        print(self.config)
        return graph
