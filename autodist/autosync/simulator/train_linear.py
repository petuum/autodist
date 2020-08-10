import os
import glob
import json
import numpy as np
from collections import OrderedDict
from os.path import expanduser
from sklearn import linear_model
from sklearn.linear_model import Ridge
from arion.simulator.utils import split_dataset

def create_features(simulation):
	runtime_coefficients = simulation['runtime_coefficients']
	var_sync_time = simulation['var_sync_time'] # dict: <var_name, [send_time_dict, receive_time_dict]>

	res = OrderedDict({
		'network_overhead': 0.0,
		'gpu_kenrel_memory_latency': 0.0,
		'constant_factor': 0.0,
		'allreduce_factor': 0.0,
	})
	for var_name, sim_time in var_sync_time.items():
		if isinstance(sim_time, list):
			# PS strategies
			send_time, receive_time = sim_time
			res['constant_factor'] += send_time['transmission'] + receive_time['transmission']
			res['network_overhead'] += send_time['network_overhead'] + receive_time['network_overhead']
			res['gpu_kenrel_memory_latency'] += send_time['gpu_kenrel_memory_latency'] + receive_time['gpu_kenrel_memory_latency']
		elif isinstance(sim_time, dict):
			# Allreduce strategy
			res['allreduce_factor'] += sim_time['transmission']
			res['network_overhead'] += sim_time['network_overhead']
			res['gpu_kenrel_memory_latency'] += sim_time['gpu_kenrel_memory_latency']
		else:
			raise ValueError

	# runtime_coefficients = {
	# 	'transmission': slowest_server_time,
	#     'network_overhead': len(worker_list),
	#     'gpu_kenrel_memory_latency': max_num_local_replica,
	#     'constant': 1.0,
	#     # possible affecting factors.
	#     'var_name': var_name,
	#     'strategy': 'ps',
	#     'local_proxy': local_proxy,
	#     'is_sparse': is_sparse,
	#     'server_list': [partition.to_dict() for partition in server_list],
	#     'worker_list': worker_list,
	#     'cpu_worker_list': cpu_worker_list,
	#     'gpu_worker_list': gpu_worker_list,
	#     'worker_num_replicas': worker_num_replicas,
	#     'max_num_local_replica': max_num_local_replica,
	# }
	# runtime_coefficients = [
	# 	runtime_coefficients['transmission'],
	# 	runtime_coefficients['network_overhead'],
	# 	runtime_coefficients['gpu_kenrel_memory_latency'],
	# ]
	return list(res.values())

def load_trial_run_data(data_dir):
	runtimes_folders = glob.glob("{}/**/runtimes".format(data_dir), recursive=True)
	X = []
	Y = []
	for runtimes_folder in runtimes_folders:
		print(runtimes_folder)
		runtimes_files = glob.glob(os.path.join(runtimes_folder, '*'))
		for runtimes_file in runtimes_files:
			# Target
			runtime = json.load(open(runtimes_file, 'r'))
			y = runtime['average']
			# Features
			simulation_file = '/'.join(runtimes_file.split('/')[:-2]) + '/simulations/' + runtimes_file.split('/')[-1]
			assert os.path.isfile(simulation_file), 'simulation_file {} does not exist'.format(simulation_file)
			simulation = json.load(open(simulation_file, 'r'))
			x = create_features(simulation)
			X.append(x)
			Y.append(y)
	return X, Y

data_dir = os.path.join(expanduser('~'), 'oceanus_simulator/lm1b-patchon')
X, Y = load_trial_run_data(data_dir)
X_train, Y_train, X_valid, Y_valid = split_dataset(X, Y)
print('X_train', X_train.shape, 'Y_train', Y_train.shape, 'X_valid', X_valid.shape, 'Y_valid', Y_valid.shape)

# Linear regression
lm = linear_model.LinearRegression()
model = lm.fit(X_train, Y_train)
predictions = lm.predict(X_valid)
print('predictions, targets: ')
pt = zip(predictions, Y_valid)
pt = sorted(pt, key=lambda x: x[1])
for p, t in pt:
	print(p, t)
train_score = lm.score(X_train, Y_train)
valid_score = lm.score(X_valid, Y_valid)
print('Linear train_score', train_score)
print('Linear valid_score', valid_score)

# Ridge regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, Y_train)
predictions = ridge.predict(X_valid)
train_score = ridge.score(X_train, Y_train)
valid_score = ridge.score(X_valid, Y_valid)
print('Ridge train_score', train_score)
print('Ridge valid_score', valid_score)


# Lasso
lasso = linear_model.Lasso(alpha=0.1)
lasso.fit(X_train, Y_train)
train_score = lasso.score(X_train, Y_train)
valid_score = lasso.score(X_valid, Y_valid)
print('Lasso train_score', train_score)
print('Lasso valid_score', valid_score)

# ElasticNet
elastic = linear_model.ElasticNet(random_state=0)
elastic.fit(X_train, Y_train)
train_score = elastic.score(X_train, Y_train)
valid_score = elastic.score(X_valid, Y_valid)
print('ElasticNet train_score', train_score)
print('ElasticNet valid_score', valid_score)
