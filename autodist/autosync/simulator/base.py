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

"""Simulator base class."""
from collections import OrderedDict

import os

from autodist.graph_item import GraphItem
from autodist.kernel.partitioner import PartitionerConfig
from autodist.resource_spec import ResourceSpec
from autodist.strategy.auto.item import VariableItem, PartItem, ResourceItem


class SimulatorBase:
    """Simulates strategies for a given graph and resource spec."""

    def __init__(self,
                 graph_item=None,
                 resource_spec=None):
        """
        Constructor for simulator base class
        Args:
            graph_item: a GraphItem object, or a path to a serialized GraphItem object.
            resource_spec: a ResourceSpec object, or a path to a resource file.
        """
        # check if it is a path
        self._graph_item = None
        # if isinstance(graph_item, GraphItem):
        #     self._graph_item = graph_item
        # elif isinstance(graph_item, str) and os.path.exists(graph_item):
        #     self._graph_item = GraphItem.deserialize(graph_item)
        # else:
        #     raise ValueError("Invalid graph_item: {}".format(graph_item))

        self._resource_spec = None
        # if isinstance(resource_spec, ResourceSpec):
        #     self._resource_spec = resource_spec
        # elif isinstance(resource_spec, str) and os.path.exists(resource_spec):
        #     self._resource_spec = ResourceSpec(resource_spec)
        # else:
        #     raise ValueError("Invalid resource_spec: {}".format(resource_spec))

    def update_graph_item(self, graph_item):
        """Change the default graph_item with this simulator."""
        if not graph_item:
            raise ValueError('Empty graph item.')
        self._graph_item = graph_item

    def update_resource_spec(self, resource_spec):
        """Change the default resource_spec with this simulator."""
        if not resource_spec:
            raise ValueError('Empty resource spec.')
        self._resource_spec = resource_spec

    def simulate(self,
                 strategy,
                 graph_item=None,
                 resource_spec=None,
                 *args,
                 **kwargs):
        """Return simulated runtime cost given (strategy, graph_item, resource_spec) tuple."""
        raise NotImplementedError()

    def inference(self, *args, **kwargs):
        """Abstract method for simulator inference."""
        raise NotImplementedError()

    def load_checkpoint(self, checkpoint):
        """
        Load a checkpoint file as weights of the simulator.

        Args:
            checkpoint: path to a checkpoint file.
        """
        raise NotImplementedError()

    # def save_checkpoint(self, model, checkpoint):
    #     """
    #     Save a trained weight as a checkpoint file.
    #
    #     Args:
    #         model: trained model.
    #         checkpoint: path where to save the checkpoint.
    #     """
    #     raise NotImplementedError()

    def preprocess(self,
                   strategy,
                   graph_item=None,
                   resource_spec=None):
        """
        Preprocess a (strategy, graph_item, resource_spec) tuple into pre-features.

        Args:
            strategy: a distribution strategy
            graph_item: optional graph_item, if not provided, the default one bundled with simulator will be used.
            resource_spec: optional resource_spec, if not provided, the default one bundled with simulator will be used.

        Returns:
            OrderedDict(): variable/part name to variable/part items.
            ResourceItem:
        """
        if not graph_item:
            if not self._graph_item:
                raise ValueError('No graph item provided.')
            else:
                graph_item = self._graph_item
        if not resource_spec:
            if not self._resource_spec:
                raise ValueError('No resource spec provided.')
            else:
                resource_spec = self._resource_spec
        if not strategy:
            raise ValueError('No strategy provided.')
        resource_item = ResourceItem(resource_spec)
        name_to_var = {var.name: var for var_op, var in graph_item.trainable_var_op_to_var.items()}

        name_to_items = OrderedDict()
        for node in strategy.node_config:
            var_name = node.var_name
            var = name_to_var[var_name]
            if node.partitioner:
                pc = PartitionerConfig(partition_str=node.partitioner)
                for i, part in enumerate(node.part_config):
                    part_item = PartItem(var, graph_item, i, pc, part)
                    name_to_items[part_item.name] = part_item
            else:
                var_item = VariableItem(var, graph_item, node)
                name_to_items[var_item.name] = var_item
        return name_to_items, resource_item
