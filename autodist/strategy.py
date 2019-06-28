from . import lib


class Strategy:
    """
    Serializable Representation
    """

    DEFAULT_STRATEGY = "AllReduce"
    subclasses = set()

    def __init__(self):
        self._def = {}

    def get_distributed_graph(self, graph):
        """
        Stateless
        """
        g = graph
        for subgraph, (distributor, config) in self._def.items():
            g = lib.getattr(distributor)(config).apply(g, target=subgraph)
        return g

    @classmethod
    def _load(cls):
        """
        Deserialize
        """
        o = cls()
        o._def = {'sub_graph1': ('PS', lib.Config()),
                  'sub_graph2': ('AR', lib.Config())}  # a deserialized dict
        return o

    @classmethod
    def _create(cls, graph, resource_spec, strategy_name):
        """
        Factory
        """
        o = cls()
        #################
        # Here:
        #   * analyze graph and resource_spec
        #   * generate strategy representation and set configurations
        #################
        # o._def = {'sub_graph1': ('PS', lib.Config()),
        #           'sub_graph2': ('AR', lib.Config())}
        o = cls.subclasses.get(strategy_name).create(graph, resource_spec)
        return o

    @classmethod
    def create(cls, graph, resource_spec, strategy_name=None):
        strategy_name = strategy_name or cls.DEFAULT_STRATEGY
        option = 1 or 2
        if option == 1:  # is master
            # Master uses factory to analyze and create strategy repr
            return cls._create(graph, resource_spec, strategy_name)
        if option == 2:  # is worker
            # Worker takes strategy repr
            return cls._load()
