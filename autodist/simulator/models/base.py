"""Strategy Simulator."""
from collections import defaultdict
import numpy as np
from enum import Enum

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape

from autodist.cluster import SSHCluster
from autodist.resource_spec import ResourceSpec
from autodist.kernel.device.resolver import DeviceResolver
from autodist.graph_item import GraphItem
from autodist.kernel.partitioner import PartitionerConfig
from autodist.proto.synchronizers_pb2 import AllReduceSynchronizer
from autodist.strategy.base import Strategy
from autodist.kernel.common.utils import get_op_name, get_consumers

from utils import resolve_device_address, get_max_num_local_replica, get_num_local_replica

GIGABITS = np.float(1e+9)
INFINITY = 1e+9
NUM_RUNS = 500



class Var:
    def __init__(self,
                 name=None,
                 is_sparse=False,
                 synchronizer=None,
                 shape=None,
                 dtype=None,
                 device=None,
                 compressor=None):
        self.name = name
        self.is_sparse = is_sparse
        self.synchronizer = synchronizer
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.compressor = compressor
        self.device = device
        self.is_partition = False

        self.original_shape = self.shape

    @property
    def var_size(self):
        size = 1
        if self.shape:
            for s in self.shape:
                size *= s
        return size

    @property
    def original_var_size(self):
        size = 1
        if self.original_shape:
            for s in self.original_shape:
                size *= s
        return size

    def size_to_transfer(self, batch_size_per_gpu=1, seq_len=1):
        if not self.is_sparse:
            return self.var_size
        else:
            if not self.shape:  # scalar
                return 1

            emb_size = 1
            if len(self.shape) > 1:
                for i in range(1, len(self.original_shape)):
                    emb_size = emb_size * self.original_shape[i]

            sparse_data_size = batch_size_per_gpu * seq_len * emb_size

            # estimate the embedding of this partition simply using a proportional formula
            ret = sparse_data_size * self.var_size / self.original_var_size
            return ret

class Partition(Var):
    def __init__(self,
                 name=None,
                 is_sparse=False,
                 synchronizer=None,
                 shape=None,
                 dtype=None,
                 device=None,
                 compressor=None,
                 part_id=0,
                 original_shape=None,
                 partition_str=None,
                 num_shards=1):
        super(Partition, self).__init__(name, is_sparse, synchronizer, shape, dtype, device, compressor)
        self.is_partition = True
        self.part_id = part_id
        self.partition_str = partition_str
        self.original_shape = original_shape
        self.num_shards = num_shards

class Resource:
    def __init__(self, cluster, device_resolver, graph_replicas, network_bandwidth, cpu_worker_list,
                 gpu_worker_list, max_num_local_replica, total_num_local_replica, worker_num_replicas):
        self.cluster=cluster
        self.device_resolver=device_resolver
        self.graph_replicas=graph_replicas
        self.network_bandwidth=network_bandwidth
        self.cpu_worker_list=cpu_worker_list
        self.gpu_worker_list=gpu_worker_list
        self.max_num_local_replica=max_num_local_replica
        self.total_num_local_replica=total_num_local_replica
        self.worker_num_replicas=worker_num_replicas

class VarType(Enum):
    SPARSE = 0
    DENSE = 1

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


class VariableHelper:
    def __init__(self, var, graph_item):
        self.var = var
        self.graph_item = graph_item
        self._var_op_name = get_op_name(var.name)
        self._grad = graph_item.var_op_name_to_grad_info[self._var_op_name][0]

    @property
    def var_type(self):
        return VarType.DENSE if isinstance(self._grad, ops.Tensor) else VarType.SPARSE

    @property
    def is_sparse(self):
        return True if self.var_type == VarType.SPARSE else False

    @property
    def is_embedding(self):
        for op in get_consumers(self.var.op):
            if op.type == "ResourceGather":
                return True
            # op = new_graph_item.graph.get_operation_by_name(
            #     ops.prepend_name_scope(op.name, ARION_TO_DELETE_SCOPE)
            # )
        return False

    @property
    def shape(self):
        if self.var.initial_value.shape.ndims:
            return self.var.initial_value.shape.as_list()
        else:
            return None

    @property
    def partitionable_axis(self):
        valid_axis = []
        if not self.shape:
            return valid_axis
        # Sparse variable can only be partition along the 0th axis
        # only sample axis for dense variables
        if self.is_sparse or self.is_embedding:
            valid_axis = [0]
            return valid_axis
        for idx, dim in enumerate(self.shape):
            if dim > 1:
                valid_axis.append(idx)
        return valid_axis

    @property
    def byte_size(self):
        return float(byte_size_load_fn(self.var))

    @property
    def dtype(self):
        return self.var.dtype


class PartHelper:
    def __init__(self, part_idx, var, pc):
        self.var = var
        self.part_idx = part_idx
        self.pc = pc

    @property
    def shape(self):
        shape = self.var.initial_value.shape.as_list()
        dim_size = shape[self.pc.axis] // self.pc.num_shards
        extras = shape[self.pc.axis] % self.pc.num_shards
        if self.part_idx < extras:
            dim_size += 1
        shape[self.pc.axis] = dim_size
        return shape

    @property
    def var_shape(self):
        return self.var.initial_value.shape.as_list()

    @property
    def byte_size(self):
        return float(byte_size_load_fn(self.var)) \
               * float(self.shape[self.pc.axis]) / float(self.var_shape[self.pc.axis])



class SimulatorBase:
    """Simulates strategies for a given graph and resource spec."""

    def __init__(self, original_graph_item_path):
        self._original_graph_item_path = original_graph_item_path
        self._original_graph_item = GraphItem.deserialize(original_graph_item_path)

    def simulate(self, strategy: Strategy, resource_spec: ResourceSpec, checkpoint: str):
        """Return simulated runtime value by feeding features to the cost model."""
        raise NotImplementedError()

    def inference(self, inputs, checkpoint):
        raise NotImplementedError()

    def load_checkpoint(self, checkpoint):
        raise NotImplementedError()

    def save_checkpoint(self, model, checkpoint):
        raise NotImplementedError()

    def create_features(self, strategy: Strategy, resource_spec: ResourceSpec):
        raise NotImplementedError()

    def extract_pre_feature(self, strategy: Strategy, resource_spec: ResourceSpec, cluster: SSHCluster,
                            device_resolver: DeviceResolver):
        resource = self.setup_resource(resource_spec, cluster, device_resolver)

        name2var = {var.name: var for var_op, var in self._original_graph_item.trainable_var_op_to_var.items()}

        meta = defaultdict()
        for node in strategy.node_config:
            var_name = node.var_name
            # for var_op, var in self._original_graph_item.trainable_var_op_to_var.items():
            #     if var.name == var_name:
            #         break
            var = name2var[var_name]
            var_helper = VariableHelper(var, self._original_graph_item)

            if node.partitioner:
                pc = PartitionerConfig(partition_str=node.partitioner)
                for i, part in enumerate(node.part_config):
                    part_helper = PartHelper(i, var, pc)
                    synchronizer = getattr(part, part.WhichOneof('synchronizer'))
                    compressor = getattr(synchronizer, 'compressor', None)
                    reduction_destination = getattr(synchronizer, 'reduction_destination', None)
                    device = resolve_device_address(reduction_destination if reduction_destination else var.device,
                                                     resource.device_resolver)

                    part_meta = Partition(name=part.var_name,
                                          is_sparse=var_helper.is_sparse,
                                          shape=part_helper.shape,
                                          dtype=var_helper.dtype,
                                          synchronizer=synchronizer,
                                          part_id=i,
                                          num_shards=pc.num_shards,
                                          partition_str=pc.partition_str,
                                          original_shape=var_helper.shape,
                                          compressor=compressor,
                                          device=device)
                    meta[part_meta.name] = part_meta
            else:
                synchronizer = getattr(node, node.WhichOneof('synchronizer'))
                compressor = getattr(synchronizer, 'compressor', None)
                reduction_destination = getattr(synchronizer, 'reduction_destination', None)
                device = resolve_device_address(reduction_destination if reduction_destination else var.device,
                                                 resource.device_resolver)

                var_meta = Var(name=var_name,
                               is_sparse=var_helper.is_sparse,
                               shape=var_helper.shape,
                               dtype=var_helper.dtype,
                               synchronizer=synchronizer,
                               compressor=compressor,
                               device=device)
                meta[var_meta.name] = var_meta
        return meta, resource

    def extract_pre_feature_legacy(self, strategy):
        """Don't use now!!!"""
        meta = defaultdict()
        for node in strategy.node_config:
            var_name = node.var_name
            for var_op, var in self._original_graph_item.trainable_var_op_to_var.items():
                if var.name == var_name:
                    break
            var_op_name = var_op.name
            var_helper = VariableHelper(var, self._original_graph_item)
            synchronizer = getattr(node, node.WhichOneof('synchronizer'))
            compressor = getattr(synchronizer, 'compressor', None)
            if compressor is not None:
                compressor = AllReduceSynchronizer.Compressor.Name(compressor)
            reduction_destinations = getattr(synchronizer, 'reduction_destinations', None)
            if not reduction_destinations or len(reduction_destinations) <= 1:
                # this variable is not partitioned
                device = reduction_destinations[0] if reduction_destinations else var.device
                var_meta = Var(name=var_name,
                               is_sparse=var_helper.is_sparse,
                               shape=var_helper.shape,
                               dtype=var_helper.dtype,
                               synchronizer=synchronizer,
                               compressor=compressor,
                               device=device)
                meta[var_meta.name] = var_meta
            else:
                # this variable is partitioned
                num_partitions = len(reduction_destinations)
                partition_list = [1] * len(var_helper.shape)
                partition_list[0] = num_partitions
                pc = PartitionerConfig(partition_list=partition_list)
                for i, device in enumerate(reduction_destinations):
                    part_helper = PartHelper(i, var, pc)
                    part_meta = Partition(name='{}/part_{}:0'.format(var_op_name, i),
                                          is_sparse=var_helper.is_sparse,
                                          shape=part_helper.shape,
                                          dtype=var_helper.dtype,
                                          synchronizer=synchronizer,
                                          part_id=i,
                                          partition_str=pc.partition_str,
                                          original_shape=var_helper.shape,
                                          compressor=compressor,
                                          device=device)
                    meta[part_meta.name] = part_meta
        return meta

    def setup_resource(self, resource_spec: ResourceSpec, cluster: SSHCluster, device_resolver: DeviceResolver):
        graph_replicas = [resolve_device_address(k, device_resolver) for k, v in resource_spec.gpu_devices]
        # bandwidth
        network_bandwidth = self.network_bandwidth(resource_spec, device_resolver)
        # Other information
        cpu_worker_list = [resolve_device_address(device, device_resolver) for device, _ in resource_spec.cpu_devices]
        gpu_worker_list = [resolve_device_address(device, device_resolver) for device, _ in resource_spec.gpu_devices]
        max_num_local_replica = get_max_num_local_replica(graph_replicas, cluster)
        total_num_local_replica = len(graph_replicas)
        worker_num_replicas = [get_num_local_replica(cpu_worker, graph_replicas, cluster)
                               for cpu_worker in cpu_worker_list]
        resource = Resource(cluster=cluster,
                            device_resolver=device_resolver,
                            graph_replicas=graph_replicas,
                            network_bandwidth=network_bandwidth,
                            cpu_worker_list=cpu_worker_list,
                            gpu_worker_list=gpu_worker_list,
                            max_num_local_replica=max_num_local_replica,
                            total_num_local_replica=total_num_local_replica,
                            worker_num_replicas=worker_num_replicas)
        return resource

    @staticmethod
    def network_bandwidth(resource_spec: ResourceSpec, device_resolver: DeviceResolver):
        """Calculates all P2P network bandwidths between nodes in the cluster."""
        devices = [device for device, _ in resource_spec.devices]
        resolved_devices = [resolve_device_address(device, device_resolver) for device, _ in resource_spec.devices]
        gpu_cpu_bw = 10000.  # hardcode for now
        network_bandwidth = {}  # key: <server, worker>
        for i in range(len(devices)):
            if resolved_devices[i] not in network_bandwidth:
                network_bandwidth[resolved_devices[i]] = {}
            for j in range(i, len(devices)):
                if resolved_devices[j] not in network_bandwidth:
                    network_bandwidth[resolved_devices[j]] = {}
                ip_i = devices[i].split(':')[0]
                ip_j = devices[j].split(':')[0]
                if ip_i != ip_j:
                    network_bandwidth[resolved_devices[i]][resolved_devices[j]] \
                        = GIGABITS * resource_spec.network_bandwidth[ip_i]    # todo: solve.
                    network_bandwidth[resolved_devices[j]][resolved_devices[i]] \
                        = GIGABITS * resource_spec.network_bandwidth[ip_j]
                else:
                    network_bandwidth[resolved_devices[i]][resolved_devices[j]] = GIGABITS * gpu_cpu_bw
                    network_bandwidth[resolved_devices[j]][resolved_devices[i]] = GIGABITS * gpu_cpu_bw

        return network_bandwidth

    @staticmethod
    def min_bandwitdh(worker_list, bandwidth):
        min_bandwidth = INFINITY
        num_workers = len(worker_list)
        for i in range(num_workers):
            for j in range(i, num_workers):
                min_bandwidth = min(min_bandwidth, bandwidth[worker_list[j]][worker_list[i]])

    @property
    def original_graph_item_path(self):
        return self._original_graph_item_path
