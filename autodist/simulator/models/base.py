"""Strategy Simulator."""
import time
from collections import defaultdict
import numpy as np

import tensorflow as tf
from tensorflow.python.client import timeline

from arion.simulator.utils import NUM_RUNS
from arion.cluster import SSHCluster
from arion.graph_item import GraphItem
from arion.kernel.device.resolver import DeviceResolver
from arion.kernel.partitioner import PartitionerConfig
from arion.proto.synchronizers_pb2 import AllReduceSynchronizer
from arion.resource_spec import ResourceSpec
from arion.strategy.base import Strategy
from arion.simulator.utils import _resolve_device_address, GIGABITS, _max_num_local_replica, _num_local_replica
from arion.strategy.random_sample_strategy import VariableHelper, PartHelper
from arion.simulator.utils import INFINITY

# tf.compat.v1.disable_eager_execution()

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

class SimulatorBase:
    """Simulates strategies for a given graph and resource spec."""

    def __init__(self, original_graph_item_path):
        self._original_graph_item_path = original_graph_item_path
        self._original_graph_item = GraphItem.deserialize(original_graph_item_path)
        # self._resource_file = resource_file
        # self._resource_spec = ResourceSpec(resource_file)
        # self._cluster = SSHCluster(self._resource_spec)
        # self._device_resolver = DeviceResolver(self._cluster)
        #
        # self._graph_replicas = [_resolve_device_address(k, self._device_resolver)
        #                         for k, v in self._resource_spec.gpu_devices]
        #
        # # bandwidth
        # self._network_bandwidth = self.network_bandwidth(self._resource_spec, self._device_resolver)
        # # Other information
        # self._cpu_worker_list = [_resolve_device_address(device, self._device_resolver)
        #                          for device, _ in self._resource_spec.cpu_devices]
        # self._gpu_worker_list = [_resolve_device_address(device, self._device_resolver)
        #                          for device, _ in self._resource_spec.gpu_devices]
        # self._max_num_local_replica = _max_num_local_replica(self._graph_replicas, self._cluster)
        # self._total_num_local_replica = len(self._graph_replicas)
        # self._worker_num_replicas = [_num_local_replica(cpu_worker, self._graph_replicas, self._cluster)
        #                              for cpu_worker in self._cpu_worker_list]

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

    def extract_pre_feature(self, strategy: Strategy, resource_spec: ResourceSpec):
        resource = self.setup_resource(resource_spec)

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
                    device = _resolve_device_address(reduction_destination if reduction_destination else var.device,
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
                device = _resolve_device_address(reduction_destination if reduction_destination else var.device,
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

    def setup_resource(self, resource_spec: ResourceSpec):
        cluster = SSHCluster(resource_spec)
        device_resolver = DeviceResolver(cluster)
        graph_replicas = [_resolve_device_address(k, device_resolver) for k, v in resource_spec.gpu_devices]
        # bandwidth
        network_bandwidth = self.network_bandwidth(resource_spec, device_resolver)
        # Other information
        cpu_worker_list = [_resolve_device_address(device, device_resolver) for device, _ in resource_spec.cpu_devices]
        gpu_worker_list = [_resolve_device_address(device, device_resolver) for device, _ in resource_spec.gpu_devices]
        max_num_local_replica = _max_num_local_replica(graph_replicas, cluster)
        total_num_local_replica = len(graph_replicas)
        worker_num_replicas = [_num_local_replica(cpu_worker, graph_replicas, cluster) for cpu_worker in cpu_worker_list]
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
        resolved_devices = [_resolve_device_address(device, device_resolver) for device, _ in resource_spec.devices]
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
                        = GIGABITS * resource_spec.network_bandwidth[ip_i]
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

    # @property
    # def resource_file(self):
    #     return self._resource_file

    @staticmethod
    def calculate_op_timings(fetches):
        # Simple implementation. Calculate averaged run time of certain steps.
        init_op = tf.compat.v1.initialize_all_variables()
        outside_times = []

        with tf.compat.v1.Session() as sess:
            sess.run(init_op)
            for i in range(NUM_RUNS):
                start = time.time()
                sess.run(fetches)
                end = time.time()
                outside_times.append(end - start)
        comp_time_in_sec = np.mean(np.array(outside_times[1:]))
        return comp_time_in_sec

    @staticmethod
    def profile_on_single_machine(fetches):
        # calculate computation time of every op
        init_op = tf.compat.v1.initialize_all_variables()
        op_name2runtime = defaultdict(list)
        outside_times = []
        all_times = []

        options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
        run_metadata = tf.compat.v1.RunMetadata()
        with tf.compat.v1.Session() as sess:
            sess.run(init_op)
            for i in range(NUM_RUNS):
                start = time.time() * 1000
                sess.run(fetches)
                end = time.time() * 1000
                outside_times.append(end - start)

                sess.run(fetches, options=options, run_metadata=run_metadata)

                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()  # necessary
                for event in fetched_timeline._chrome_trace._events:
                    # print('\n')
                    # print(list(event.keys()))
                    # for key in list(event.keys()):
                    #     print(key, event[key])
                    if 'dur' in event:
                        op_name2runtime[event['args']['name']].append(event['dur'])
                    # todo: to be more accurate, add tid (thread/lanes id)

        mean_outside_time = np.mean(np.array(outside_times[1:]))
        print('mean outside_times: ', mean_outside_time)
        print(outside_times)
        # print('average all_times: ', np.mean(np.array(all_times)))

        op_name2meanruntime = {}
        for op_name, runtimes in op_name2runtime.items():
            runtimes = np.array(runtimes)
            if len(runtimes) > 1:  # Do not compute operations that only run once for all steps.
                mean = np.mean(np.array(runtimes[1:]))
                op_name2meanruntime[op_name] = mean
                print(op_name, mean)
                # print(op_name2runtime[op_name])

        total_op_time = sum([mean_runtime for op_name, mean_runtime in op_name2meanruntime.items()])
        print('total_op_time', total_op_time / 1000.)
        # total_op_time = [sum([runtime[i] for op_name, runtime in op_name2runtime.items()])
        # for i in range(self.num_runs)]
        # print('total_op_time', np.mean(np.array(total_op_time)), total_op_time)

        return mean_outside_time

    # @staticmethod
    # def _calculate_op_timings(graph_item: GraphItem):
    #     """
    #     Given a graph, calculates an expected running time for each (op, input_size) pair.
    #
    #     Args:
    #         graph_item (GraphItem): The input graph.
    #
    #     Returns:
    #         Dict mapping (op, input_size) to time.
    #     """
    #     all_ops = {}
    #     for op in graph_item.graph.get_operations():
    #         input_shapes = tuple((tuple(inp.shape.dims) for inp in op.inputs))
    #         op_type = op.type
    #         all_ops[(op_type, input_shapes)] = ops.Graph()
    #
    #     for ((op, shape), graph) in all_ops.items():
    #         with graph.as_default():
    #             getattr(tensorflow.raw_ops, op)
