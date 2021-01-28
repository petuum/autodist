import tensorflow as tf
from tensorflow.python.framework import device_spec

from autodist.kernel.device.resolver import DeviceResolver


DTYPE2BITS = {
	tf.float16: 16,
	"tf.float16": 16,
	"<dtype: 'float16'>": 16,
	tf.float32: 32,
	'tf.float32': 32,
	"<dtype: 'float32'>": 32,
	"<dtype: 'float32_ref'>": 32,
	tf.float64: 64,
	'tf.float64': 64,
	"<dtype: 'float64'>": 64,
	tf.bfloat16: 16,
	'tf.bfloat16': 16,
	"<dtype: 'bfloat16'>": 16,
	tf.complex64: 64,
	'tf.complex64': 64,
	"<dtype: 'complex64'>": 64,
	tf.complex128: 128,
	'tf.complex128': 128,
	"<dtype: 'complex128'>": 128,
	tf.int8: 8,
	'tf.int8': 8,
	"<dtype: 'int8'>": 8,
	tf.uint8: 8,
	'tf.uint8': 8,
	"<dtype: 'uint8'>": 8,
	tf.uint16: 16,
	'tf.uint16': 16,
	"<dtype: 'uint16'>": 16,
	tf.uint32: 32,
	'tf.uint32': 32,
	"<dtype: 'uint32'>": 32,
	tf.uint64: 64,
	'tf.uint64': 64,
	"<dtype: 'uint64'>": 64,
	tf.int16: 16,
	'tf.int16': 16,
	"<dtype: 'int16'>": 16,
	tf.int32: 32,
	'tf.int32': 32,
	"<dtype: 'int32'>": 32,
	tf.int64: 64,
	'tf.int64': 64,
	"<dtype: 'int64'>": 64,
	tf.bool: 1,
	'tf.bool': 1,
	"<dtype: 'bool'>": 1,
	tf.string: 1,  # todo: confirm
	'tf.string': 1,  # todo: confirm
	"<dtype: 'string'>": 1,  # todo: confirm
	tf.qint8: 8,
	'tf.qint8': 8,
	"<dtype: 'qint8'>": 8,
	tf.quint8: 8,
	'tf.quint8': 8,
	"<dtype: 'quint8'>": 8,
	tf.qint16: 16,
	'tf.qint16': 16,
	"<dtype: 'qint16'>": 16,
	tf.quint16: 16,
	'tf.quint16': 16,
	"<dtype: 'quint16'>": 16,
	tf.qint32: 32,
	'tf.qint32': 32,
	"<dtype: 'qint32'>": 32,
	tf.resource: 0,  # its tensor shape is either [] or [None] todo: confirm
	'tf.resource': 0,  # its tensor shape is either [] or [None] todo: confirm
	"<dtype: 'resource'>": 0,  # its tensor shape is either [] or [None] todo: confirm
}

def get_dtype_bits(dtype):
	return DTYPE2BITS[dtype] if dtype in DTYPE2BITS else DTYPE2BITS[str(dtype)]


def get_dense_var_bits(size, dtype):
	return size * get_dtype_bits(dtype)



def get_sparse_var_bits(size):
	# same size of values, indices, dense_shape
	return size * (get_dtype_bits(tf.float32) + 2 * get_dtype_bits(tf.int64)) \
		   + 2 * get_dtype_bits(tf.int64)



def resolve_device_address(device: str, device_resolver: DeviceResolver):
	# change real ip address to /job:worker/task:0
	if not device:
		return device
	parts = device.split(':')
	if parts and parts[0] in device_resolver._address_to_tasks:
		resolved_device = device_resolver._address_to_tasks[parts[0]][0]
		resolved = '/job:{}/task:{}/device:'.format(resolved_device['job'], resolved_device['task'])
		resolved = resolved + ':'.join(parts[-2:])
		return resolved
	else:
		raise ValueError("cannot resolve device: {} using device_resolver: {}".format(
			device, device_resolver._address_to_tasks))


def resolved_devices_on_diff_machine(device1, device2):
	# e.g., '/job:worker/task:1/device:CPU:0', '/job:worker/task:1/GPU:0'
	node1 = ':'.join(device1.split('/')[:-1])
	node2 = ':'.join(device2.split('/')[:-1])
	return node1 != node2
	

def get_max_num_local_replica(replicas, cluster):
	replica_devices = {device_spec.DeviceSpecV2.from_string(r) for r in replicas}
	replica_hosts = {cluster.get_address_from_task(d.job, d.task) for d in replica_devices}
	max_num_local_replica = 0
	for host in replica_hosts:
		num_local_replica = sum(1 for d in replica_devices
								if cluster.get_address_from_task(d.job, d.task) == host)
		max_num_local_replica = max(max_num_local_replica, num_local_replica)
	return max_num_local_replica


def get_num_local_replica(host, replicas, cluster):
	# host: e.g., '/job:worker/task:0/device:CPU:0'
	replica_devices = {device_spec.DeviceSpecV2.from_string(r) for r in replicas}
	host_device = device_spec.DeviceSpecV2.from_string(host)
	num_local_replica = sum(1 for d in replica_devices
							if cluster.get_address_from_task(d.job, d.task) ==
							cluster.get_address_from_task(host_device.job, host_device.task))
	return num_local_replica


