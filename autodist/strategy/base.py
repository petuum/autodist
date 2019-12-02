"""Strategy Base."""

import os
from datetime import datetime

from autodist.const import DEFAULT_SERIALIZATION_DIR, Env
from autodist.proto import strategy_pb2
from autodist.resource_spec import ResourceSpec


class StrategyBuilder:
    """
    A base builder for various strategies.

    Returns:
        [type]: [description]
    """

    def __init__(self, item, resource_spec: ResourceSpec):
        self._item = item
        self._resource_spec = resource_spec

    @classmethod
    def all_subclasses(cls):
        """Get all subclasses recursively as a set."""
        subclasses = set()
        for subclass in cls.__subclasses__():
            subclasses.add(subclass)
            subclasses.update(subclass.all_subclasses())
        return subclasses

    @classmethod
    def get_subclasses(cls):
        """Get the mapping between strategy name and strategy classes."""
        return {c.__name__: c for c in cls.all_subclasses()}

    @classmethod
    def build(cls, item, resource_spec, strategy_name):
        """
        Build strategy representation instance.

        TODO: change the method name
        """
        if strategy_name not in cls.get_subclasses():
            strategy_name = 'Auto'

        o = cls.get_subclasses()[strategy_name](item, resource_spec)
        strategy = o._build()  # pylint: disable=protected-access
        return strategy

    def _build(self):
        pass

    @classmethod
    def load_strategy(cls):
        """Load serialized strategy."""
        strategy_id = os.environ[Env.AUTODIST_STRATEGY_ID.name]
        o = Strategy.deserialize(strategy_id)
        return o


class Strategy:
    """A wrapper around a Strategy Protocol Buffer."""

    def __init__(self, strategy=None):
        self._strategy = strategy or strategy_pb2.Strategy()
        if strategy is None:
            self._strategy.id = datetime.utcnow().strftime('%Y%m%dT%H%M%SM%f')

    @property
    def id(self):
        """Strategy's ID."""
        return self._strategy.id

    @property
    def path(self):
        """Strategy's Path."""
        return self._strategy.path

    @property
    def node_config(self):
        """Strategy's Node Config."""
        return self._strategy.node_config

    @node_config.setter
    def node_config(self, value):
        """Set this Strategy's Node Config."""
        if self._strategy.node_config is not value:
            # TODO: is this the best way?
            del self._strategy.node_config[:]
            self._strategy.node_config.extend(value)

    @property
    def graph_config(self):
        """Strategy's Graph Config."""
        return self._strategy.graph_config

    @graph_config.setter
    def graph_config(self, value):
        """Set this Strategy's Graph Config."""
        self._strategy.graph_config = value

    def copy(self):
        """Create a copy of this strategy."""
        other_strategy = strategy_pb2.Strategy()
        other_strategy.CopyFrom(self._strategy)
        return Strategy(strategy=other_strategy)

    def __str__(self):
        return self._strategy.__str__()

    def serialize(self, path=None):
        """Serialize this strategy and write it to disk."""
        if path is None:
            os.makedirs(DEFAULT_SERIALIZATION_DIR, exist_ok=True)
            path = os.path.join(DEFAULT_SERIALIZATION_DIR, self._strategy.id)

        with open(path, "wb+") as f:
            f.write(self._strategy.SerializeToString())

        self._strategy.path = path

    @classmethod
    def deserialize(cls, strategy_id=None, path=None):
        """Deserialize the strategy."""
        if path is None:
            assert strategy_id is not None
            path = os.path.join(DEFAULT_SERIALIZATION_DIR, strategy_id)
        return cls(strategy=strategy_pb2.Strategy.ParseFromString(open(path, 'r')))


class StrategyCompiler:
    """Strategy Compiler."""

    def __init__(self):
        self._device_resolver = None

    def set_device_resolver(self, resolver):
        """Add a device resolver to resolve devices in the strategy."""
        self._device_resolver = resolver
        return self

    def _resolve_devices(self, strategy):
        s = strategy.copy()
        for n in s.node_config:
            synchronizer = getattr(n, n.WhichOneof('synchronizer'))
            if hasattr(synchronizer, 'reduction_destinations'):
                d = synchronizer.reduction_destinations
                synchronizer.reduction_destinations[:] = self._device_resolver(d)
        d = s.graph_config.replicas
        s.graph_config.replicas[:] = self._device_resolver(d)
        return s

    def compile(self, strategy):
        """Compile the strategy."""
        if self._device_resolver:
            strategy = self._resolve_devices(strategy)
        return strategy
