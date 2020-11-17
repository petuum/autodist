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

"""Poseidon StrategyBuilder."""

from tensorflow.python.framework import tensor_shape

from autodist.strategy.base import Strategy, StrategyBuilder
from autodist.proto import strategy_pb2, synchronizers_pb2, compressor_pb2
from autodist.kernel.common.utils import get_op_name


class Poseidon(StrategyBuilder):
    """
    PS StrategyBuilder with Greedy Load Balancing.

    The Load Balancing is determined by total memory
    usage for storing variables, i.e. we always assign
    a variable to the current lowest-memory-usage
    Parameter Server.
    """

    #pylint: disable=too-many-arguments
    def __init__(self, batch_size, local_proxy_variable=False, sync=True, staleness=0, 
                 broadcast_spec='NCCL', compressor='SFBCompressor'):
        if batch_size < 1:
            raise ValueError('The batch_size must be greater than zero.')
        self._batch_size = batch_size
        self._local_proxy_variable = local_proxy_variable
        self._sync = sync
        self._staleness = staleness
        if self._staleness > 0:
            assert self._sync, 'If staleness is positive, sync has to be set true.'
        self.broadcast_spec = broadcast_spec
        self.compressor = compressor
        self.loads = {}
        super().__init__()

    def build(self, graph_item, resource_spec):
        """Generate the Strategy."""
        expr = Strategy()

        # get each variable, generate variable synchronizer config
        expr.graph_config.replicas.extend([k for k, v in resource_spec.gpu_devices])
        for k, v in resource_spec.node_cpu_devices.items():
            if k not in resource_spec.node_gpu_devices:
                expr.graph_config.replicas.extend(v)

        num_servers = resource_spec.num_cpus
        num_workers = resource_spec.num_gpus

        # find all variables
        variables = graph_item.trainable_var_op_to_var.values()
        reduction_device_names = [k for k, _ in resource_spec.cpu_devices]
        self.loads = {ps: 0.0 for ps in reduction_device_names}

        # Mark each variable to be synchronized with a Parameter Server
        for var in variables:
            op_name = get_op_name(var.name)
            shape = get_op_shape(var)
            if op_name == 'sequential/dense/kernel' and ((2 * self._batch_size * (num_workers - 1) *
                                                          int(shape[0] + shape[1])) <= (2 * int(shape[0]) * 
                                                                                        int(shape[1]) * 
                                                                                        (num_servers + 
                                                                                         num_workers - 2)
                                                                                        / num_servers)):
                node_config = self._gen_sfb_node_config(var.name, broadcast_spec=self.broadcast_spec, 
                                                        compressor=self.compressor)
            else:
                node_config = self._gen_ps_node_config(var, self._local_proxy_variable, self._sync, 
                                                       self._staleness)
            expr.node_config.append(node_config)

        return expr

    def _gen_ps_node_config(self, var, local_proxy_variable, sync, staleness):
        """
        Creates a NodeConfig specifying synchronization with Parameter Servers.

        Args:
            var (Variable): The variable to generate a config for.

        Returns:
            strategy_pb2.Strategy.Node: the config for the node.
        """
        min_ps = min(self.loads, key=self.loads.get)
        self.loads[min_ps] += byte_size_load_fn(var)

        node = strategy_pb2.Strategy.Node()
        node.var_name = var.name
        node.PSSynchronizer.reduction_destination = min_ps
        node.PSSynchronizer.local_replication = local_proxy_variable
        node.PSSynchronizer.sync = sync
        node.PSSynchronizer.staleness = staleness
        return node

    @staticmethod
    def _gen_sfb_node_config(var_name, group=0, broadcast_spec="NCCL", compressor="SFBCompressor"):
        """
        Creates a NodeConfig specifying synchronization with Parameter Servers.

        Args:
            var (Variable): The variable to generate a config for.

        Returns:
            strategy_pb2.Strategy.Node: the config for the node.
        """
        node = strategy_pb2.Strategy.Node()
        node.var_name = var_name
        node.SFBSynchronizer.spec = synchronizers_pb2.SFBSynchronizer.Spec.Value(broadcast_spec)
        node.compressor.type = compressor_pb2.Compressor.Type.Value(compressor)
        node.SFBSynchronizer.group = group
        return node


def get_op_shape(op):
    """Get number of elements in a "variable" op."""
    shape = op.get_shape()
    if not shape.is_fully_defined():
        # Due to legacy behavior, scalar "Variable" ops have output Tensors that
        # have unknown shape when the op is created (and hence passed to this
        # load function for placement), even though the scalar shape is set
        # explicitly immediately afterward.
        shape = tensor_shape.TensorShape(op.get_attr("shape"))
    shape.assert_is_fully_defined()
    return shape


def byte_size_load_fn(op):
    """
    Load function that computes the byte size of a single-output `Operation`.

    Copied (with modifications) from tensorflow.contrib.training.python.training.device_setter.

    This is intended to be used with `"Variable"` ops, which have a single
    `Tensor` output with the contents of the variable.  However, it can also be
    used for calculating the size of any op that has a single output.

    Intended to be used with `GreedyLoadBalancingStrategy`.

    Args:
      op: An `Operation` with a single output, typically a "Variable" op.

    Returns:
      The number of bytes in the output `Tensor`.

    Raises:
      ValueError: if `op` does not have a single output, or if the shape of the
        single output is not fully-defined.
    """
    elem_size = op.dtype.size
    shape = op.get_shape()
    if not shape.is_fully_defined():
        # Due to legacy behavior, scalar "Variable" ops have output Tensors that
        # have unknown shape when the op is created (and hence passed to this
        # load function for placement), even though the scalar shape is set
        # explicitly immediately afterward.
        shape = tensor_shape.TensorShape(op.get_attr("shape"))
    shape.assert_is_fully_defined()
    return shape.num_elements() * elem_size
