# Copyright 2020 Petuum, Inc. All Rights Reserved.
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

"""PS StrategyBuilder."""

from autodist.strategy.base import Strategy, StrategyBuilder
from autodist.proto import strategy_pb2


class PS(StrategyBuilder):
    """
    PS StrategyBuilder.

    Generates a Strategy that synchronizes every variable
    using Parameter Servers. Each variable is only assigned
    to one Parameter Server.
    """

    def __init__(self, local_proxy_variable=False, sync=True, staleness=0):
        self._local_proxy_variable = local_proxy_variable
        self._sync = sync
        self._staleness = staleness
        if self._staleness > 0:
            assert self._sync, 'If staleness is positive, sync has to be set true.'

    def build(self, graph_item, resource_spec):
        """Build PS strategy."""
        expr = Strategy()

        # get each variable, generate variable synchronizer config
        expr.graph_config.replicas.extend([k for k, v in resource_spec.gpu_devices])
        for k, v in resource_spec.node_cpu_devices.items():
            if k not in resource_spec.node_gpu_devices:
                expr.graph_config.replicas.extend(v)

        # find all variables
        variables = graph_item.trainable_var_op_to_var.values()
        reduction_device_name = [k for k, _ in resource_spec.cpu_devices][0]

        # Mark each variable to be synchronized with a Parameter Server
        node_config = [self._gen_ps_node_config(var.name, reduction_device_name, self._local_proxy_variable,
                                                self._sync, self._staleness)
                       for var in variables]
        expr.node_config.extend(node_config)
        return expr

    @staticmethod
    def _gen_ps_node_config(var_name, reduction_destination, local_proxy_variable, sync, staleness):
        """
        Creates a NodeConfig specifying synchronization with Parameter Servers.

        Args:
            var_name (str): The name of the variable.
            reduction_destinations (Iter[str]): The location of the parameter servers.

        Returns:
            strategy_pb2.Strategy.Node: the config for the node.
        """
        node = strategy_pb2.Strategy.Node()
        node.var_name = var_name
        node.PSSynchronizer.reduction_destination = reduction_destination
        node.PSSynchronizer.local_replication = local_proxy_variable
        node.PSSynchronizer.sync = sync
        node.PSSynchronizer.staleness = staleness
        return node
