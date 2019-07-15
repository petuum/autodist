"""Strategy Base."""

from datetime import datetime
import os
import yaml

from autodist.const import DEFAULT_SERIALIZATION_DIR
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
        strategy.serialize()
        return strategy

    def _build(self):
        pass

    @classmethod
    def load_strategy(cls):
        """Load serialized strategy."""
        strategy_id = os.environ['AUTODIST_STRATEGY_ID']
        o = Strategy.deserialize(strategy_id)
        return o


class Strategy:
    """Strategy representation."""

    def __init__(self):
        self._id = datetime.utcnow().strftime('%Y%m%dT%H%M%SM%f')
        self.node_config = {}
        self.graph_config = {}

    def as_dict(self):
        """Strategy representation as dict"""
        return {
            '_id': self._id,
            'node_config': self.node_config,
            'graph_config': self.graph_config
        }

    def serialize(self):
        """
        Serialize the strategy.

        TODO: Maybe protobuf later
        """
        os.makedirs(DEFAULT_SERIALIZATION_DIR, exist_ok=True)
        path = os.path.join(DEFAULT_SERIALIZATION_DIR, self._id)
        yaml.safe_dump(self.as_dict(), stream=open(path, 'w+'))

    @classmethod
    def deserialize(cls, strategy_id):
        """Deserialize the strategy."""
        path = os.path.join(DEFAULT_SERIALIZATION_DIR, strategy_id)
        o = cls()
        o.__dict__.update(yaml.safe_load(open(path, 'r')))
        return o

    def __str__(self):
        return str(self.as_dict())
