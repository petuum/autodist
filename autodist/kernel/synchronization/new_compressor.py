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

"""Gradient Compressors for All-Reduce."""
from abc import ABC, abstractmethod
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.ops import collective_ops, math_ops

from autodist.kernel.common.utils import get_op_name
from autodist.proto import compressor_pb2

#from tensorflow.python.ops import array_ops, collective_ops, linalg_ops, math_ops, random_ops
#from autodist.kernel.synchronization.collective_key import get_collective_keys
#from autodist.utils import logging


class CollectiveOpsConfig:
    """Config for using Collective Ops."""

    group_size: int
    group_key: str
    instance_key: str
    merge_op: str
    final_op: str


class Compressor(ABC):
    """
    Wraps CollectiveOps.All_Reduce with compression and decompression for network efficiency.

    This means that it only wraps gradient transmission for AllReduce
    synchronized variables, not PS ops or other ops like network reads.
    """

    def __init__(self, var_name):
        var_op_name = get_op_name(var_name)
        self.var_op_name = var_op_name
        self.num_workers = None
        self.num_replicas = None
        self.worker_device = None
        self.worker_id = None
        self.is_chief = None
        self.all_canonical_replica_devices = None

    @abstractmethod
    def reduce(self, tensor: Tensor, conf: CollectiveOpsConfig):
        """
        Compress, reduce, and decompress a given tensor.

        Args:
            tensor (Tensor): the Tensor to reduce.
            conf (CollectiveOpsConfig): the config for Collective Ops.

        Returns:
            Reduced Tensor
        """

    @abstractmethod
    def _compress(self, tensor: Tensor):
        """
        Compress a given tensor.

        Args:
            tensor (Tensor): the Tensor to compress.

        Returns:
            Tensor
        """

    @abstractmethod
    def _decompress(self, compressed_tensor: Tensor):
        """
        Decompress a given tensor.

        Args:
            compressed_tensor (Tensor): the Tensor to decompress.

        Returns:
            Tensor, Context
        """

    @staticmethod
    def _all_reduce(tensor: Tensor, conf: CollectiveOpsConfig):
        """
        Using CollectiveOps, AllReduce the given tensor.

        Args:
            tensor (Tensor): the tensor to all-reduce
            conf (CollectiveOpsConfig): the config for CollectiveOps

        Returns:
            The All-Reduced Tensor
        """
        return collective_ops.all_reduce(tensor, **conf.__dict__)


    def in_graph_compress(self, graph_item, var_name):
        item = graph_item
        var_op_name = get_op_name(var_name)
        master_op_name = ops.prepend_name_scope(var_op_name, replica_prefix(0))
        grad, _, _ = graph_item.var_op_name_to_grad_info[master_op_name]
        with item.graph.as_default():
            self._share_initializer(item, var_op_name, master_replica=0)
            if isinstance(grad, ops.IndexedSlices):
                self._compress_sparse_gradients(item, var_op_name)
        return item


    def in_graph_decompress(self, graph_item, var_name):
        item = graph_item
        var_op_name = get_op_name(var_name)
        master_op_name = ops.prepend_name_scope(var_op_name, replica_prefix(0))

    def _compress_sparse_gradients(item, var_op_name):



    def _share_initializer(self, graph_item, var_op_name, master_replica=0):
    """Share the initializers of all replica variables to use initializer on replica=master_replica."""
    # find the initial value of the var on master_replica
    master_var_op = graph_item.graph.get_operation_by_name(
        ops.prepend_name_scope(var_op_name, replica_prefix(master_replica)))
    master_var = graph_item.trainable_var_op_to_var[master_var_op]
    master_init_tensor = graph_item.graph.get_tensor_by_name(master_var.initial_value.name)
    master_init_op = master_init_tensor.op
    # set the device of the init ops to reside on the chief device
    master_init_device = device_spec.DeviceSpecV2.from_string(master_init_op.device) \
        .replace(task=0)
    master_init_op._set_device_from_string(master_init_device.to_string())

    for i in range(0, self.num_replicas):
        if i == master_replica:
            continue
        var_op = graph_item.graph.get_operation_by_name(
            ops.prepend_name_scope(var_op_name, replica_prefix(i)))
        var = graph_item.trainable_var_op_to_var[var_op]
        init_op = graph_item.graph.get_tensor_by_name(var.initial_value.name).op
        init_assign_op = get_consumers(init_op)[0]
        init_assign_op._update_input(1, master_init_tensor)

    def assign_cluster_information(self,
                                   num_workers,
                                   num_replicas,
                                   worker_device,
                                   worker_id,
                                   canonical_replica_devices,
                                   is_chief=False):
        """Store cluster information in the synchronizer."""
        self.num_workers = num_workers
        self.num_replicas = num_replicas
        self.worker_device = worker_device  # local worker device
        self.worker_id = worker_id  # local worker id
        self.all_canonical_replica_devices = canonical_replica_devices
        self.is_chief = is_chief
        return self

    @classmethod
    def create(cls, name, *args, **kwargs):
        """
        Create new Compressor instance given subclass name.

        Args:
            name: Name of the Compressor subclass (e.g. NoneCompressor).
            *args: Any args for the subclass constructor.
            **kwargs: Any kwargs for the subclass constructor.

        Returns:
            Compressor
        """
        name = compressor_pb2.Compressor.Type.Name(compressor_value)
        subclass = next(subclass for subclass in cls._get_subclasses() if subclass.__name__ == name)
        return subclass(*args, **kwargs)

    @classmethod
    def _get_subclasses(cls):
        return set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in c._get_subclasses()])


# pylint: disable=abstract-method
class CompressorEF(Compressor, ABC):
    """A Compressor with Error Feedback."""

    def __init__(self, var_op_name):
        self.error = None
        super().__init__(var_op_name)

    def reduce(self, tensor: Tensor, conf: CollectiveOpsConfig):
        """
        Compress, reduce, and decompress a given tensor.

        Args:
            tensor (Tensor): the Tensor to reduce.
            conf (CollectiveOpsConfig): the config for Collective Ops.

        Returns:
            Reduced Tensor
        """
        if self.error is not None:
            tensor += self.error
        compressed_tensor = self._compress(tensor)
        self.error = tensor - self._decompress(compressed_tensor)
        reduced = self._all_reduce(compressed_tensor, conf)
        return self._decompress(reduced)


class NoneCompressor(Compressor):
    """An identity Compressor."""

    def reduce(self, tensor: Tensor, conf: CollectiveOpsConfig):
        """
        Compress, reduce, and decompress a given tensor.

        Args:
            tensor (Tensor): the Tensor to reduce.
            conf (CollectiveOpsConfig): the config for Collective Ops.

        Returns:
            Reduced Tensor
        """
        return self._all_reduce(tensor, conf)

    def _compress(self, tensor: Tensor):
        return tensor

    def _decompress(self, compressed_tensor: Tensor):
        return compressed_tensor


class HorovodCompressor(Compressor):
    """Implements Horovod's Compression."""

    def __init__(self, var_op_name):
        self.dtype = None
        super().__init__(var_op_name)

    def reduce(self, tensor: Tensor, conf: CollectiveOpsConfig):
        """
        Compress, reduce, and decompress a given tensor.

        Args:
            tensor (Tensor): the Tensor to reduce.
            conf (CollectiveOpsConfig): the config for Collective Ops.

        Returns:
            Reduced Tensor
        """
        compressed_tensor = self._compress(tensor)
        reduced = self._all_reduce(compressed_tensor, conf)
        return self._decompress(reduced)

    def _compress(self, tensor: Tensor):
        self.dtype = tensor.dtype
        tensor_compressed = tensor
        if tensor.dtype.is_floating:
            # Only allow compression from other floating point types
            # TODO: dtypes.float16 if using TF2.1.x (errors in 2.0)
            tensor_compressed = math_ops.cast(tensor, dtypes.float32)
        return tensor_compressed

    def _decompress(self, compressed_tensor: Tensor):
        return math_ops.cast(compressed_tensor, self.dtype)


class HorovodCompressorEF(CompressorEF, HorovodCompressor):  # This works because of Method Resolution Order
    """Horovod's Compression but with Error Feedback."""


# class PowerSGDCompressor(CompressorEF):
#     """An implementation of the PowerSGD compression algorithm (arxiv.org/abs/1905.13727)."""

#     def __init__(self, var_op_name, rank=1):
#         self.rank = rank
#         self.og_shape, self.ndims, self.new_shape, self.compressor = None, None, None, None
#         super().__init__(var_op_name)

#     def reduce(self, tensor: Tensor, conf: CollectiveOpsConfig):
#         """
#         Compress, reduce, and decompress a given tensor.

#         Args:
#             tensor (Tensor): the Tensor to reduce.
#             conf (CollectiveOpsConfig): the config for Collective Ops.

#         Returns:
#             Reduced Tensor
#         """
#         if self.og_shape is None:
#             self.og_shape = tensor.shape
#             self.ndims = len(self.og_shape)

#         # Check if rank 1 tensor (this shouldn't be called with sparse tensors)
#         # Just reduce it if it is, no need to compress
#         if self._is_1d:
#             return self._all_reduce(tensor, conf)

#         logging.info(f"Compressing tensor {tensor.name} (var {self.var_op_name}) with shape {tensor.shape}")
#         if self.ndims > 2:
#             tensor = array_ops.reshape(tensor, [self.og_shape[0], -1])

#         if self.compressor is None:
#             self.new_shape = array_ops.shape_v2(tensor)
#             self.compressor = random_ops.random_normal([self.new_shape[1], self.rank])

#         if self.error is not None:
#             tensor += self.error

#         compressed_tensor = self._compress(tensor)
#         self.error = tensor - self._decompress(compressed_tensor)

#         # all reduce mean p
#         reduced = self._all_reduce(compressed_tensor, conf)
#         reduced = self._orthogonalize(reduced)

#         # update compressor
#         self.compressor = math_ops.matmul(tensor, reduced, transpose_a=True)
#         conf.instance_key = get_collective_keys().get_instance_key(self.var_op_name + "/compressor")
#         self.compressor = self._all_reduce(self.compressor, conf)
#         return array_ops.reshape(self._decompress(reduced), self.og_shape) \
#             if self.ndims > 2 else self._decompress(reduced)

#     def _compress(self, tensor: Tensor):
#         return math_ops.matmul(tensor, self.compressor)

#     def _decompress(self, compressed_tensor: Tensor):
#         return math_ops.matmul(compressed_tensor, self.compressor, transpose_b=True)

#     @property
#     def _is_1d(self):
#         return self.ndims <= 1 or (
#             self.ndims == 2 and any(d == 1 for d in self.og_shape)
#         )

#     @staticmethod
#     def _orthogonalize(matrix):
#         _, m = matrix.shape
#         for i in range(m):
#             v = matrix[:, i]
#             v /= linalg_ops.norm_v2(v)
#             v = array_ops.expand_dims_v2(v, 1)

#             begin, rest = matrix[:, :i], matrix[:, (i + 1):]
#             rest -= math_ops.matmul(v, rest, transpose_a=True) * v
#             matrix = array_ops.concat([begin, v, rest], 1)
#         return matrix
