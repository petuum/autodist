# Copyright 2020 Petuum. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains Strategy-related definitions."""

import os
from abc import ABC, abstractmethod
from datetime import datetime

from autodist.const import DEFAULT_SERIALIZATION_DIR
from autodist.graph_item import GraphItem
from autodist.kernel.common.utils import get_op_name
from autodist.proto import strategy_pb2
from autodist.resource_spec import ResourceSpec


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

        self._strategy.path = path

        with open(path, "wb+") as f:
            f.write(self._strategy.SerializeToString())

    @classmethod
    def deserialize(cls, strategy_id=None, path=None):
        """Deserialize the strategy."""
        if path is None:
            assert strategy_id is not None
            path = os.path.join(DEFAULT_SERIALIZATION_DIR, strategy_id)
        with open(path, 'rb') as f:
            data = f.read()
        new_strategy = strategy_pb2.Strategy()
        new_strategy.ParseFromString(data)
        return cls(strategy=new_strategy)


class StrategyBuilder(ABC):
    """A builder interface for strategies."""

    @abstractmethod
    def build(self, graph_item: GraphItem, resource_spec: ResourceSpec) -> Strategy:
        """
        Build strategy representation instance with a graph item and a resource spec.

        Args:
            graph_item (graph_item.GraphItem): the graph for which to develop a strategy
            resource_spec (ResourceSpec): resource information

        Returns:
            (Strategy) A strategy representation instance.
        """
        raise NotImplementedError


class StrategyCompiler:
    """
    Strategy Compiler.

    Currently, this just entails resolving device attributes,
    but this can be easily modified to do more in the future.
    """

    def __init__(self, graph_item):
        self._graph_item = graph_item
        self._device_resolver = None

    def set_device_resolver(self, resolver):
        """Add a device resolver to resolve devices in the strategy."""
        self._device_resolver = resolver
        return self

    def _resolve_reduction_destination(self, node):
        synchronizer = getattr(node, node.WhichOneof('synchronizer'))
        if hasattr(synchronizer, 'reduction_destination'):
            d = synchronizer.reduction_destination
            synchronizer.reduction_destination = self._device_resolver(d)

    def _resolve_devices(self, strategy):
        s = strategy.copy()
        for n in s.node_config:
            if n.partitioner:
                # meaning this var is going to be partitioned
                for part in n.part_config:
                    self._resolve_reduction_destination(part)
            else:
                self._resolve_reduction_destination(n)
        d = s.graph_config.replicas
        s.graph_config.replicas[:] = self._device_resolver(d)
        return s

    def _prune_nodes(self, strategy):
        # Prune the nodes without stateful updates
        s = strategy.copy()
        s.node_config = [n for n in strategy.node_config
                         if get_op_name(n.var_name) in self._graph_item.var_op_name_to_grad_info]
        return s

    def compile(self, strategy):
        """Compile the strategy."""
        strategy = self._prune_nodes(strategy)
        if self._device_resolver:
            strategy = self._resolve_devices(strategy)
        return strategy
