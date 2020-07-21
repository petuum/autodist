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

"""Predefined simulator with linear model."""

import pickle as pkl

import tensorflow as tf
from tensorflow.python.eager import context

from autodist.proto.synchronizers_pb2 import PSSynchronizer, AllReduceSynchronizer
from autodist.resource_spec import ResourceSpec
from autodist.simulator.base import SimulatorBase
from autodist.simulator.utils import _resolved_devices_on_diff_machine, \
	get_dense_var_bits, get_sparse_var_bits
from autodist.strategy.base import Strategy


class PredefinedSimulator(SimulatorBase):
	"""Simulates strategies for a given graph and resource spec."""

	def __init__(self,
				 graph_item=None,
				 resource_spec=None,
				 batch_size=1,
				 seq_len=1,
				 get_coef=True,
				 checkpoint=None):

		super(PredefinedSimulator, self).__init__(original_graph_item_path=original_graph_item_path)

		print("It's using predefined simulator. batch_size_per_gpu is {}".format(batch_size))
		self._fetches = fetches
		self._batch_size_per_gpu = batch_size
		self._seq_len = seq_len
		self._get_coef = get_coef
		self._checkpoint = checkpoint
		self._weights = None
		with context.eager_mode():
			if self._checkpoint:
				self._weights = self.load_checkpoint(self._checkpoint)

	def simulate(self, strategy: Strategy, resource_spec: ResourceSpec, checkpoint=None):
		"""Return simulated runtime value."""
		inputs = self.create_features(strategy, resource_spec)
		with context.eager_mode():
			cost = self.inference(inputs, checkpoint)
		return cost

	def inference(self, inputs, checkpoint=None):
		if checkpoint is not None:
			weights = self.load_checkpoint(checkpoint)
		elif self._weights is not None:
			weights = self._weights
		else:
			raise ValueError("No checkpoint provided in either initialization or inference.")

		if not isinstance(inputs, tf.Tensor):
			inputs = tf.reshape(tf.convert_to_tensor(inputs), [1, len(inputs)])

		if len(weights) == 4:
			W0, b0, W, b = weights
			inputs = tf.nn.elu(tf.matmul(inputs, W0) + b0)
			cost = tf.matmul(inputs, W) + b
		elif len(weights) == 2:
			W, b = weights
			cost = tf.matmul(inputs, W) + b
		else:
			raise ValueError
		return cost

	def load_checkpoint(self, checkpoint=None):
		if checkpoint is None:
			if self._checkpoint is not None:
				checkpoint = self._checkpoint
			else:
				raise ValueError("checkpoint is None: {}".format(checkpoint))
		self._weights = pkl.load(open(checkpoint, 'rb'))
		# self._weights = json.load(open(checkpoint, 'r'))
		print("Load checkpoint: ")
		print(self._weights)
		return self._weights

	def save_checkpoint(self, model, checkpoint):
		pkl.dump(model, open(checkpoint, 'wb'))
		self._checkpoint = checkpoint
		self._weights = model

	def create_features_v0(self, strategy: Strategy, resource_spec: ResourceSpec):
		var_sync_time, vars, resource = self.predefined_sync_time(strategy, resource_spec)

		# Add up sync time per device to find the slowest server time.
		feature_keys = ['transmission', 'network_overhead', 'gpu_kernel_memory_latency']
		device_ps_sync_time = {}
		var_ar_sync_time = {}
		for var_name, sync_time in var_sync_time.items():
			if isinstance(vars[var_name].synchronizer, PSSynchronizer):
				device = vars[var_name].device
				if device not in device_ps_sync_time:
					device_ps_sync_time[device] = {key: 0.0 for key in feature_keys}
				for key in feature_keys:
					device_ps_sync_time[device][key] += sync_time[0][key] + sync_time[1][key]

			else: # AllReduce
				if var_name not in var_ar_sync_time:
					var_ar_sync_time[var_name] = {key: 0.0 for key in feature_keys}
				for key in feature_keys:
					var_ar_sync_time[var_name][key] += sync_time[key]

		max_device_ps_sync_time = {key: 0.0 for key in feature_keys}
		sum_device_ps_sync_time = {key: 0.0 for key in feature_keys}
		sum_var_ar_sync_time = {key: 0.0 for key in feature_keys}
		for key in feature_keys:
			max_device_ps_sync_time[key] = max([d[key] for d in device_ps_sync_time.values()] or [0.0])
			sum_device_ps_sync_time[key] = sum([d[key] for d in device_ps_sync_time.values()] or [0.0])
			sum_var_ar_sync_time[key] = sum([d[key] for d in var_ar_sync_time.values()] or [0.0])

		feat = [max_device_ps_sync_time[key] for key in feature_keys] \
		       + [sum_device_ps_sync_time[key] for key in feature_keys] \
		       + [sum_var_ar_sync_time[key] for key in feature_keys]

		return feat

	def create_features(self, strategy: Strategy, resource_spec: ResourceSpec):
		# var_sync_time, vars, resource = self.predefined_sync_time(strategy, resource_spec)

		vars, resource = self.preprocess(strategy=strategy, resource_spec=resource_spec)

		feature_keys = ['transmission', 'network_overhead', 'gpu_kernel_memory_latency']
		device_ps_sync_time = {}
		group_ar_sync_time = {}

		for var_name, var in vars.items():
			if isinstance(var.synchronizer, PSSynchronizer):
				sync_time = self.var_ps_time(var, resource)
				device = vars[var_name].device
				if device not in device_ps_sync_time:
					device_ps_sync_time[device] = {key: 0.0 for key in feature_keys}
				for key in feature_keys:
					device_ps_sync_time[device][key] += sync_time[0][key] + sync_time[1][key]
			elif isinstance(var.synchronizer, AllReduceSynchronizer):
				sync_time = self.var_ar_time(var, resource)
				var_group = sync_time['group']
				if var_group not in group_ar_sync_time:
					group_ar_sync_time[var_group] = {key: 0.0 for key in feature_keys}
				for key in feature_keys:
					group_ar_sync_time[var_group][key] += sync_time[key]
			else:
				raise ValueError('{}'.format(type(var.synchronizer)))

		max_device_ps_sync_time = {key: 0.0 for key in feature_keys}
		sum_device_ps_sync_time = {key: 0.0 for key in feature_keys}
		max_group_ar_sync_time = {key: 0.0 for key in feature_keys}
		sum_group_ar_sync_time = {key: 0.0 for key in feature_keys}
		for key in feature_keys:
			max_device_ps_sync_time[key] = max([d[key] for d in device_ps_sync_time.values()] or [0.0])
			sum_device_ps_sync_time[key] = sum([d[key] for d in device_ps_sync_time.values()] or [0.0])
			max_group_ar_sync_time[key] = max([d[key] for d in group_ar_sync_time.values()] or [0.0])
			sum_group_ar_sync_time[key] = sum([d[key] for d in group_ar_sync_time.values()] or [0.0])

		feat = [max_device_ps_sync_time[key] for key in feature_keys] \
		       + [sum_device_ps_sync_time[key] for key in feature_keys] \
		       + [max_group_ar_sync_time[key] for key in feature_keys] \
		       + [sum_group_ar_sync_time[key] for key in feature_keys]

		return feat

	def predefined_sync_time(self, strategy, resource_spec):
		""" graph_item: transformed graph item """
		vars, resource = self.preprocess(strategy=strategy, resource_spec=resource_spec)
		# Compute synchronization time for every var
		var_sync_time = {}
		for var_name, var in vars.items():
			if isinstance(var.synchronizer, PSSynchronizer):
				var_sync_time[var_name] = self.var_ps_time(var, resource)
			elif isinstance(var.synchronizer, AllReduceSynchronizer):
				var_sync_time[var_name] = self.var_ar_time(var, resource)
			else:
				raise ValueError('{}'.format(type(var.synchronizer)))
		return var_sync_time, vars, resource

	def var_ps_time(self, var, resource, network_overhead=0.0, gpu_kernel_memory_latency=0.0):
		"""Compute synchronization time of a variable in PS strategy."""
		def _helper(worker_list, worker_num_replicas=None):
			if worker_num_replicas is None:
				worker_num_replicas = [1.0] * len(worker_list)

			this_server_time = 0
			# network transfer: sum up all workers time. equals to the time cost of this server.
			# TODO(Hao): didn't consider any parallelization among partitions
			for k, worker in enumerate(worker_list):
				if _resolved_devices_on_diff_machine(var.device, worker):
					if var.is_sparse:
						this_worker_size = get_sparse_var_bits(var_size_to_transfer) * worker_num_replicas[k]
					else:
						this_worker_size = get_dense_var_bits(var_size_to_transfer, var.dtype)
					this_server_time += this_worker_size / resource.network_bandwidth[var.device][worker]

			if self._get_coef:
				return {
					'transmission': this_server_time,
					'network_overhead': len(worker_list),
					'gpu_kernel_memory_latency': resource.max_num_local_replica,
					'constant': 1.0,
					# possible affecting factors.
					'var_name': var.name,
					'strategy': 'ps',
					'local_proxy': var.synchronizer.local_replication,
					'is_sparse': var.is_sparse,
					'size_to_transfer': var_size_to_transfer,
					'dtype': str(var.dtype),
					# 'server_list': [partition.to_dict() for partition in server_list],
					'worker_list': worker_list,
					'cpu_worker_list': resource.cpu_worker_list,
					'gpu_worker_list': resource.gpu_worker_list,
					'worker_num_replicas': worker_num_replicas,
					'max_num_local_replica': resource.max_num_local_replica,
					'is_ps': True,
				}
			else:
				return this_server_time + len(worker_list) * network_overhead + \
					   gpu_kernel_memory_latency * resource.max_num_local_replica

		var_size_to_transfer = var.size_to_transfer(batch_size_per_gpu=self._batch_size_per_gpu,
													seq_len=self._seq_len)

		if var.is_sparse:
			send_time = _helper(resource.cpu_worker_list, worker_num_replicas=resource.worker_num_replicas)
			receive_time = _helper(resource.gpu_worker_list)
		else:
			send_time = _helper(resource.cpu_worker_list)
			if var.synchronizer.local_replication:
				receive_time = _helper(resource.cpu_worker_list)
			else:
				receive_time = _helper(resource.gpu_worker_list)

		return send_time, receive_time

	def var_ar_time(self, var, resource, network_overhead=0.0, gpu_kernel_memory_latency=0.0):
		"""Compute synchronization time of a variable in AR strategy."""
		worker_list = resource.cpu_worker_list
		num_workers = len(worker_list)
		min_bandwidth = None
		for i in range(num_workers):
			for j in range(i, num_workers):
				if min_bandwidth is None:
					min_bandwidth = resource.network_bandwidth[worker_list[j]][worker_list[i]]
				else:
					min_bandwidth = min(min_bandwidth, resource.network_bandwidth[worker_list[j]][worker_list[i]])

		# Compressor
		if var.compressor == "PowerSGDCompressor" or var.compressor == 3:
			rank = 10  # currently using default value. So hardcode here. # todo: confirm
			# assume var must be a dense variable.
			og_shape = var.shape
			ndims = len(og_shape)
			if ndims <= 1:  # no compress
				size_to_transfer = var.size_to_transfer(batch_size_per_gpu=self._batch_size_per_gpu,
														seq_len=self._seq_len)
			else:
				if ndims > 2:
					n = og_shape[0]
					m = 1
					for s in og_shape[1:]:
						m *= s  # tensor's shape (n, m)
				else:
					n, m = og_shape[0], og_shape[1]
				size_to_transfer = n * rank + m * rank
			dtype = tf.float32
		elif var.compressor == "HorovodCompressorEF" or var.compressor == "HorovodCompressor"  \
				or var.compressor == 2  or var.compressor == 1:
			size_to_transfer = var.size_to_transfer(batch_size_per_gpu=self._batch_size_per_gpu,
													seq_len=self._seq_len)
			dtype = tf.float32
		elif var.compressor == "NoneCompressor" or var.compressor == 0:
			size_to_transfer = var.size_to_transfer(batch_size_per_gpu=self._batch_size_per_gpu,
													seq_len=self._seq_len)
			dtype = var.dtype
		else:
			raise ValueError('Compressor does not exist: {}'.format(var.compressor))

		# todo: chunk_size
		# AllReduce communication time
		# time = 2 * (num_workers - 1) * get_dense_var_bits(size_to_transfer, dtype) / (min_bandwidth * num_workers)
		time = get_dense_var_bits(size_to_transfer, dtype) / min_bandwidth

		if self._get_coef:
			return {
				'transmission': time,
				'network_overhead': 1,  # len(worker_list),
				'gpu_kernel_memory_latency': resource.max_num_local_replica,
				'constant': 1.0,
				# possible affecting factors.
				'var_name': var.name,
				'group': var.synchronizer.group,
				'strategy': 'allreduce',
				'is_sparse': False,
				# 'chunk_size': chunk_size,
				'spec': 'NCCL',  # default
				'compressor': var.compressor,
				'worker_list': worker_list,
				'num_workers': num_workers,
				'size_to_transfer': size_to_transfer,
				'dtype': str(dtype),
				'min_bandwidth': min_bandwidth,
				'max_num_local_replica': resource.max_num_local_replica,
				'is_ps': False,
			}
		else:
			return time + network_overhead * len(worker_list) \
			       + gpu_kernel_memory_latency * resource.max_num_local_replica



	# @staticmethod
	# def var_ps_time(var_name, is_sparse, local_proxy, server_list, cpu_worker_list, gpu_worker_list,
	#				 max_num_local_replica, worker_num_replicas, network_bandwidth, get_coef,
	#				 network_overhead=0.0, gpu_kernel_memory_latency=0.0):
	#	 """Compute synchrinzation time of a variable in PS strategy."""
	#
	#	 def _helper(worker_list, worker_num_replicas=None):
	#		 if worker_num_replicas is None:
	#			 worker_num_replicas = [1.0] * len(worker_list)
	#		 # Compute the slowest server
	#		 slowest_server_time = 0
	#		 for j, server in enumerate(server_list):
	#			 if server.size_to_transfer == 0:
	#				 continue
	#			 # network transfer: sum up all workers time. equals to the time cost of this server.
	#			 this_server_time = 0
	#			 for k, worker in enumerate(worker_list):
	#				 if _resolved_devices_on_diff_machine(server.device, worker):
	#					 if is_sparse:
	#						 this_worker_size = get_sparse_var_bits(server.size_to_transfer) * worker_num_replicas[k]
	#					 else:
	#						 this_worker_size = get_dense_var_bits(server.size_to_transfer, server.dtype)
	#					 this_server_time += this_worker_size / network_bandwidth[server.device][worker]
	#			 slowest_server_time = max(slowest_server_time, this_server_time)
	#
	#		 if get_coef:
	#			 return {
	#				 'transmission': slowest_server_time,
	#				 'network_overhead': len(worker_list),
	#				 'gpu_kernel_memory_latency': max_num_local_replica,
	#				 'constant': 1.0,
	#				 # possible affecting factors.
	#				 'var_name': var_name,
	#				 'strategy': 'ps',
	#				 'local_proxy': local_proxy,
	#				 'is_sparse': is_sparse,
	#				 'server_list': [partition.to_dict() for partition in server_list],
	#				 'worker_list': worker_list,
	#				 'cpu_worker_list': cpu_worker_list,
	#				 'gpu_worker_list': gpu_worker_list,
	#				 'worker_num_replicas': worker_num_replicas,
	#				 'max_num_local_replica': max_num_local_replica,
	#			 }
	#		 else:
	#			 return slowest_server_time + len(worker_list) * network_overhead + \
	#					gpu_kernel_memory_latency * max_num_local_replica
	#
	#	 if is_sparse:
	#		 send_time = _helper(cpu_worker_list, worker_num_replicas=worker_num_replicas)
	#		 receive_time = _helper(gpu_worker_list)
	#	 else:
	#		 send_time = _helper(cpu_worker_list)
	#		 if local_proxy:
	#			 receive_time = _helper(cpu_worker_list)
	#		 else:
	#			 receive_time = _helper(gpu_worker_list)
	#
	#	 if get_coef:
	#		 # return {key: send_time[key]+receive_time[key] for key in send_time.keys()}
	#		 return send_time, receive_time
	#	 else:
	#		 return send_time, receive_time
