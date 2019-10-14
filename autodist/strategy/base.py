"""Strategy Base."""

import os
from datetime import datetime
from copy import deepcopy
import yaml

from autodist.const import DEFAULT_SERIALIZATION_DIR, Env
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
    def get_subclasses(cls):
        """Get all strategy builders."""
        return {c.__name__: c for c in cls.__subclasses__()}

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
    """Strategy representation."""

    def __init__(self):
        self._id = datetime.utcnow().strftime('%Y%m%dT%H%M%SM%f')
        self.node_config = {}
        self.graph_config = {}

        self.path = ''

    def get_id(self):
        """Return the strategy id."""
        return self._id

    def as_dict(self):
        """Strategy representation as dict."""
        return {
            '_id': self._id,
            'node_config': self.node_config,
            'graph_config': self.graph_config
        }

    @classmethod
    def from_dict(cls, d):
        """Create a new Strategy instance from the serialized dict."""
        o = cls()
        o.__dict__.update(d)
        return o

    def serialize(self):
        """
        Serialize the strategy.

        TODO: Maybe protobuf later
        """
        path = os.path.join(DEFAULT_SERIALIZATION_DIR, self._id)
        yaml.safe_dump(self.as_dict(), stream=open(path, 'w+'))
        self.path = path

    @classmethod
    def deserialize(cls, strategy_id):
        """Deserialize the strategy."""
        path = os.path.join(DEFAULT_SERIALIZATION_DIR, strategy_id)
        return cls.from_dict(yaml.safe_load(open(path, 'r')))

    def __str__(self):
        return str(self.as_dict())

    def __repr__(self):
        return self.__str__()

    def copy(self):
        """Return a deepcopy of the strategy."""
        return self.from_dict(deepcopy(self.as_dict()))


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
            if 'reduction_destinations' in s.node_config[n]['synchronizer']['config']:
                d = s.node_config[n]['synchronizer']['config']['reduction_destinations']
                s.node_config[n]['synchronizer']['config']['reduction_destinations'] = self._device_resolver(d)
        d = s.graph_config['replicas']
        s.graph_config['replicas'] = self._device_resolver(d)
        return s

    def compile(self, strategy):
        """Compile the strategy."""
        if self._device_resolver:
            strategy = self._resolve_devices(strategy)
        return strategy
