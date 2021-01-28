import sys
import os
import numpy as np
import tensorflow as tf
from os.path import expanduser
import tqdm
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import argparse
import importlib
import glob 
import json

from tensorflow.python.eager import context
import tensorflow_ranking as tfr

from autodist.strategy.base import Strategy
from models.predefined_simulator import PredefinedSimulator
from autodist.cluster import SSHCluster
from autodist.resource_spec import ResourceSpec
from autodist.kernel.device.resolver import DeviceResolver
np.random.seed(110)



RankingLossKeys = {
	# Names for the ranking based loss functions.
	'pairwise_hinge_loss': tfr.losses.RankingLossKey.PAIRWISE_HINGE_LOSS,
	'pairwise_logistic_loss': tfr.losses.RankingLossKey.PAIRWISE_LOGISTIC_LOSS,
	'pairwise_soft_zero_one_loss': tfr.losses.RankingLossKey.PAIRWISE_SOFT_ZERO_ONE_LOSS,
	'softmax_loss': tfr.losses.RankingLossKey.SOFTMAX_LOSS,
	'sigmoid_cross_entropy_loss': tfr.losses.RankingLossKey.SIGMOID_CROSS_ENTROPY_LOSS,
	'mean_squared_loss': tfr.losses.RankingLossKey.MEAN_SQUARED_LOSS,
	'list_mle_loss': tfr.losses.RankingLossKey.LIST_MLE_LOSS,
	'approx_ndcg_loss': tfr.losses.RankingLossKey.APPROX_NDCG_LOSS,
}


def split_dataset(inputs, shuffle=True, train_ratio=0.7, test_ratio=0.15):
	assert isinstance(inputs, list)
	nb_elements = len(inputs)
	nb_samples = len(inputs[0])
	n_train = int(nb_samples * train_ratio)
	n_test = int(nb_samples * test_ratio)
	shuffled = []
	train = []
	valid = []
	test = []

	if shuffle:
		random_indices = np.random.permutation(list(range(nb_samples)))
		for i in range(nb_elements):
			shuffled_i = [inputs[i][j] for j in random_indices]
			train.append(shuffled_i[:n_train])
			valid.append(shuffled_i[n_train:-n_test])
			test.append(shuffled_i[-n_test:])
	else:
		for i in range(nb_elements):
			train.append(inputs[i][:n_train])
			valid.append(inputs[i][n_train:-n_test])
			test.append(inputs[i][-n_test:])

	return train, valid, test


class TFRIterator:
	def __init__(self, X, Y, list_size, batch_size, split, baseline=0.0, scale=1.0):
		assert len(X) > 0, 'data: {}'.format(len(X))
		self.X = X
		self.Y = Y
		self.list_size = list_size
		self.baseline = baseline
		self.scale = scale
		self.batch_size = batch_size
		self.split = split
		self.n = len(X)
		self.num_examples = self.get_num_examples()
		print('Split: {},\tnumber of samples: {},\tnumber of examples: {},\tmin of y: {}'.format(
			split, len(X), self.num_examples, self.get_min_y()))

	def get_min_y(self):
		return np.min(self.Y)

	def get_num_examples(self):
		n_examples = 1
		for i in range(self.list_size):
			n_examples *= (len(self.X) -1)
		return n_examples

	def get_next(self):
		xs = [[] for _ in range(self.list_size)]
		ys = []
		for i in range(self.batch_size):
			y =[]
			for j in range(self.list_size):
				ri = np.random.randint(self.n)
				rx = self.X[ri]
				ry = self.Y[ri]
				xs[j].append(np.array(rx, dtype=np.float32))
				y.append(ry)
				assert ry * self.scale - self.baseline > 0, '{}, {}, {}'.format(ry, self.scale, self.baseline)
			ys.append(y)
		xs = [np.array(xx, dtype=np.float32) for xx in xs]
		ys = np.array(ys, dtype=np.float32)
		if self.split == 'train': # normalize y as its used for loss weights.
			ys = (ys * self.scale - self.baseline)

		return xs + [ys]


def load_from_folders_offline(simulation_dirs):
	print('simulation_dirs', simulation_dirs)
	X = []
	Y = []
	for simulation_dir in simulation_dirs:
		x, y = load_from_one_folder_offline(simulation_dir)
		if len(x) == 0:
			print('Simulation folder does not have files: {}, skipping it.'.format(simulation_dir))
			continue
		Y.append(y)
		X.extend(x)

	Y = np.concatenate(Y, axis=0)
	miny = np.min(Y)
	assert len(X) == len(Y)
	return X, Y


def load_from_one_folder_offline(simulation_dir):
	runtime_files = glob.glob(os.path.join(simulation_dir, 'runtimes/*'), recursive=False)
	resource_file = os.path.join(simulation_dir, 'resource_spec.yml')

	print("Searched runtime files: {}".format(len(runtime_files)))
	X = []
	Y = []
	for runtime_file in runtime_files:
		strategy_file = runtime_file.replace("runtimes/", 'strategies/')
		if not os.path.exists(strategy_file) or not os.path.isfile(strategy_file):
			print('strategy_file does not exist: {}.'.format(strategy_file))
			continue
		X.append((strategy_file, resource_file))
		runtime = json.load(open(runtime_file, 'r'))
		y = runtime['average']
		Y.append(y)
	Y = np.array(Y, dtype=np.float)
	print('Data points:{}, simulation_dir: {}'.format(len(X), simulation_dir))
	return X, Y


def main(args, sim_model_params):

	data_dir = sim_model_params['data_dir']
	original_graph_item_path = sim_model_params['original_graph_item_path']
	batch_size = sim_model_params['batch_size']
	ranking_loss_key = sim_model_params['ranking_loss_key']
	learning_rate = sim_model_params['learning_rate']
	list_size = sim_model_params['list_size']
	baseline = sim_model_params['baseline']
	scale = sim_model_params['scale']
	save_dir = sim_model_params['save_dir']
	save_prefix = sim_model_params['save_prefix']
	do_train = sim_model_params['do_train']
	do_test = sim_model_params['do_test']
	checkpoint = sim_model_params['checkpoint']
	model_version = sim_model_params['model_version']

	# Create simulator
	simulator = PredefinedSimulator(original_graph_item_path, batch_size=sim_model_params['model_batch_size'],
	                                seq_len=sim_model_params['model_seq_len'])

	# Create features
	strategy_resource_files, Y = load_from_folders_offline(data_dir)
	print("Createing features...")
	X = []
	prev_resource_file = None
	with context.graph_mode():
		for strategy_file, resource_file in tqdm.tqdm(strategy_resource_files):
			# For one folder with a common resource spec file, we load it only once to avoid costly computation.
			if prev_resource_file is None or resource_file != prev_resource_file:
				prev_resource_file = resource_file
				resource_spec = ResourceSpec(resource_file)
				cluster = SSHCluster(resource_spec)
				device_resolver = DeviceResolver(cluster)
			# x = simulator.create_features(Strategy.deserialize(strategy_file), ResourceSpec(resource_file))
			x = simulator.create_features(Strategy.deserialize(strategy_file), resource_spec, cluster, device_resolver)
			X.append(x)
	X = np.array(X, dtype=np.float)
	print("Finished createing features.")

	# Create model
	W = tf.Variable(tf.random.uniform([args.hidden_dim, 1]), name='W', dtype=tf.float32)
	b = tf.Variable(0.0, name='b', dtype=tf.float32)
	if model_version == 'v2':
		W0 = tf.Variable(tf.random.uniform([args.hidden_dim, args.hidden_dim]), name='W0', dtype=tf.float32)
		b0 = tf.Variable(0.0, name='b0', dtype=tf.float32)
	loss_fn = tfr.losses.make_loss_fn(RankingLossKeys[ranking_loss_key])
	major_version, _, _ = tf.version.VERSION.split('.')
	if major_version == '1':
		optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	else:
		optimizer = tf.optimizers.Adam(learning_rate)

	def forward(xs):
		rs = []
		for x in xs:
			if model_version == 'v2':
				x = tf.nn.elu(tf.matmul(x, W0) + b0)
			r = tf.matmul(x, W) + b
			rs.append(r)
		r = tf.concat(rs, axis=1, name='logits')
		return r

	@tf.function
	def train_steps(inputs_iterator, total_steps, loss_fn):

		def train_step(input, loss_fn):
			with tf.GradientTape() as tape:
				logits = forward(input[:-1])
				loss = loss_fn(labels=input[-1], logits=logits, features={})
				vs = [W0, b0, W, b] if model_version == 'v2' else [W, b]
				gradients = tape.gradient(loss, vs)
				train_op = optimizer.apply_gradients(zip(gradients, vs))
			pred = tf.squeeze(tf.argmax(logits, axis=1))
			labels = tf.squeeze(tf.argmax(input[-1], axis=1))
			acc = tf.equal(pred, labels)
			return loss, acc

		losses = []
		accs = []
		for step in range(total_steps):
			l, a = train_step(inputs_iterator.get_next(), loss_fn)
			losses.append(l)
			accs.append(a)
		return losses, accs

	@tf.function
	def eval_step(input):
		logits = forward(input[:-1])
		preds = tf.squeeze(tf.argmax(logits, axis=1))
		labels = tf.squeeze(tf.argmax(input[-1], axis=1))
		acc = tf.equal(preds, labels)
		return acc, labels, preds, input[-1], logits

	def eval_steps(iterator, total_test_steps):
		test_acc = []
		test_preds = []
		test_labels = []
		test_logits = []
		test_scores = []
		for step in range(total_test_steps):
			acc, labels, preds, scores, logits = eval_step(iterator.get_next())
			test_acc.append(acc)
			test_labels.append(labels)
			test_preds.append(preds)
			test_scores.append(scores)
			test_logits.append(logits)
		test_acc = tf.concat(test_acc, axis=0)
		test_acc = tf.cast(test_acc, tf.float32)
		avg_test_acc = tf.math.reduce_mean(test_acc)
		test_labels = tf.concat(test_labels, axis=0)
		test_preds = tf.concat(test_preds, axis=0)
		test_scores = tf.concat(test_scores, axis=0)
		test_logits = tf.concat(test_logits, axis=0)
		return avg_test_acc, test_acc, test_labels, test_preds, test_scores, test_logits

	if do_train:
		train_set, valid_set, test_set = split_dataset([X, Y], shuffle=True, train_ratio=0.7, test_ratio=0.15)
		X_train, Y_train = train_set
		X_valid, Y_valid = valid_set
		X_test, Y_test = test_set
		inputs_iterator = TFRIterator(X=X_train, Y=Y_train, list_size=list_size, batch_size=batch_size, split='train',
		                              baseline=baseline, scale=scale)
		valid_iterator = TFRIterator(X=X_valid, Y=Y_valid, list_size=list_size, batch_size=batch_size, split='valid')
		test_iterator = TFRIterator(X=X_test, Y=Y_test, list_size=list_size, batch_size=batch_size, split='test')
		total_train_steps = max(1, min(inputs_iterator.get_num_examples() // batch_size, 100))
		total_valid_steps = max(1, valid_iterator.get_num_examples() // batch_size)
		total_test_steps = max(1, test_iterator.get_num_examples() // batch_size)
		print("Total train steps per epoch: {}".format(total_train_steps))
		print("Total valid steps per epoch: {}".format(total_valid_steps))
		print("Total test steps: {}".format(total_test_steps))


		print("\nTrain model...")
		losses = []
		for epoch in range(args.epochs):
			loss, acc = train_steps(inputs_iterator, total_train_steps, loss_fn)
			losses.extend(loss)
			avgloss = sum(losses) / float(len(losses))
			print('Step: {}, avgloss: {:.5f}'.format(epoch, avgloss))
			if (epoch+1) % args.eval_every_epochs == 0:
				print("\nEvaluate on valid set...")
				avg_valid_acc, *_= eval_steps(valid_iterator, total_valid_steps)
				print('avg_valid_acc: {}'.format(avg_valid_acc.numpy()))
				print("Evaluate on test set...")
				avg_test_acc, *_= eval_steps(test_iterator, total_test_steps)
				print('avg_test_acc: {}\n'.format(avg_test_acc.numpy()))
				print('W', W.numpy())
				print('b', b.numpy())

				if (epoch+1) % args.save_every_epochs == 0:
					if not os.path.exists(save_dir):
						os.mkdir(save_dir)
					checkpoint = '{}/{}_{}_{:.5f}_{:.5f}'.format(save_dir, save_prefix, epoch+1,
					                                             avg_valid_acc, avg_test_acc)
					print("Save to {}".format(checkpoint))
					simulator.save_checkpoint([W0, b0, W, b] if model_version == 'v2' else [W, b], checkpoint)

	elif do_test:
		print("Load from {}".format(checkpoint))
		weights = simulator.load_checkpoint(checkpoint)
		if model_version == 'v2' and len(weights) == 4:
			W0, b0, W, b = weights
		elif model_version == 'v1' and len(weights) == 2:
			W, b = weights
		else:
			raise ValueError

		test_iterator = TFRIterator(X=X, Y=Y, list_size=list_size, batch_size=batch_size, split='test')
		total_test_steps = max(1, test_iterator.get_num_examples() // batch_size)
		print("\nEvaluate on test set...")
		avg_test_acc, test_acc, test_labels, test_preds, test_scores, test_logits = eval_steps(test_iterator, total_test_steps)
		for i, labels, preds, scores, logits in zip(range(100), test_labels, test_preds, test_scores, test_logits):
			print('labels', labels.numpy(), 'preds', preds.numpy(), 'scores', scores.numpy(), 'logits', logits.numpy())
		print('avg_test_acc', avg_test_acc.numpy())

		test_iterator_single = TFRIterator(X=X, Y=Y, list_size=1, batch_size=len(X), split='test')
		print("\nEvaluate each example in test set...")
		avg_test_acc, test_acc, test_labels, test_preds, test_scores, test_logits = eval_steps(test_iterator_single, 1)
		for i, labels, preds, scores, logits in zip(range(100), test_labels, test_preds, test_scores, test_logits):
			print('labels', labels.numpy(), 'preds', preds.numpy(), 'scores', scores.numpy(), 'logits', logits.numpy())
		test_logits = sorted(list(test_logits.numpy()))
		top_10_persent = test_logits[:int(len(test_logits)*0.1)]
		print('top_10_persent', top_10_persent)
		print('top_10_persent threshold', top_10_persent[-1])
		print('test_logits', test_logits)




def get_args_parser():
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument("-ms", "--model_to_sim", default='bert', type=str, help="")
	parser.add_argument("-sc", "--simulation_config", default='config', type=str, help="")
	parser.add_argument("-hd", "--hidden_dim", default=12, type=int, help="")
	parser.add_argument("-es", "--epochs", default=100, type=int, help="")
	parser.add_argument("-ee", "--eval_every_epochs", default=10, type=int, help="")
	parser.add_argument("-se", "--save_every_epochs", default=100, type=int, help="")
	return parser


if __name__ == "__main__":
	parser = argparse.ArgumentParser("Predefined simularot training script", parents=[get_args_parser()])
	args = parser.parse_args()

	module = importlib.import_module(args.simulation_config)  # import module from str
	simulation_params = getattr(module, "simulation_params")

	main(args, simulation_params[args.model_to_sim])


