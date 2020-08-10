import sys
import os
import numpy as np
import tensorflow as tf
from os.path import expanduser
import tqdm

from tensorflow.python.eager import context
import tensorflow_ranking as tfr

from arion.strategy.base import Strategy
from arion.resource_spec import ResourceSpec
from arion.simulator import utils
from arion.simulator.models.predefined_simulator import PredefinedSimulator
from arion.simulator.utils import RankingLossKeys

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

model_params = {
	'ncf_large_adam_dense': {
		'model_batch_size': 256,
		'model_seq_len': 1,
		'data_dir': [
			# '/home/hao.zhang/oceanus_cost_model_training_data/ncf/ncf_large_adam_dense_orca_16_random_search_ar_only',
			# '/home/hao.zhang/oceanus_cost_model_training_data/ncf/ncf-5-11-20/ncf_large_adam_dense_ar_only_by_chunk',
			# '/home/hao.zhang/oceanus_cost_model_training_data/ncf/ncf-5-11-20/ncf_large_adam_dense_ar_only_christy',
			# '/home/hao.zhang/oceanus_cost_model_training_data/ncf/ncf-5-11-20/ncf_large_adam_dense_ar_only_ordered_balanced',
			# '/home/hao.zhang/oceanus_cost_model_training_data/ncf/ncf-5-11-20/ncf_large_adam_dense_ar_only_ordered_balanced_12_12',
			# '/home/hao.zhang/oceanus_cost_model_training_data/ncf/ncf-5-11-20/ncf_large_adam_dense_ar_only_ordered_balanced_20_50',
			# '/home/hao.zhang/oceanus_cost_model_training_data/ncf/ncf-5-11-20/ncf_large_adam_dense_sorted_christy_ordered_balanced_30_50',
			'/home/hao.zhang/oceanus_cost_model_training_data/ncf/ncf-5-11-20/ncf_large_adam_dense_sorted_christy_ordered_balanced_30_50_2',
			# '/home/hao.zhang/oceanus_cost_model_training_data/ncf/ncf_large_adam_dense_orca_16',
			# '/home/hao.zhang/oceanus_cost_model_training_data/ncf/ncf_large_adam_dense_orca_16_random_search_christy_lb',
			# '/home/hao.zhang/oceanus_cost_model_training_data/ncf/ncf_large_adam_dense_orca_16_random_search_christy_lb_ps_only',
			# '/home/hao.zhang/oceanus_cost_model_training_data/ncf/ncf_large_adam_dense_orca_16_real_random',
			# '/home/hao.zhang/oceanus_cost_model_training_data/ncf/ncf_large_adam_dense_orca_8',
			# '/home/hao.zhang/oceanus_cost_model_training_data/ncf/ncf_large_adam_dense_random_search_orca_4',
			# '/home/hao.zhang/oceanus_cost_model_training_data/ncf/ncf_large_adam_dense_random_search_orca_16',
			# '/home/hao.zhang/oceanus_cost_model_training_data/ncf/ncf_large_adam_dense_random_search_linear_cost_model',
			# '/home/hao.zhang/oceanus_cost_model_training_data/ncf/ncf_large_adam_dense_random_search_linear_cost_model_2',
			# '/home/hao.zhang/oceanus_cost_model_training_data/ncf/ncf_large_adam_dense_random_search_linear_cost_model_2',
			# '/home/hao.zhang/oceanus_cost_model_training_data/ncf/ncf-5-9-20/ncf_large_adam_dense_orca_16_christy_lb_if_partition_lb_linear_cost_ps_only',
			# '/home/hao.zhang/oceanus_cost_model_training_data/ncf/ncf-5-9-20/ncf_large_adam_dense_orca_16_christy_lb_if_partition_lb_num_partition_2_32_linear_cost_ps_only',
			# '/home/hao.zhang/oceanus_cost_model_training_data/ncf/ncf-5-9-20/ncf_large_adam_dense_random_search_christy_lb_ps_only_if_partition_lb_ranknet_simulator_2',
			# '/home/hao.zhang/oceanus_cost_model_training_data/ncf/ncf-5-9-20/ncf_large_adam_dense_random_search_christy_lb_ps_only_ranknet_simulator',
			# '/home/hao.zhang/oceanus_cost_model_training_data/ncf/ncf_large_adam_dense_g3.4.25.1_g3.4.25.2',
			# '/home/hao.zhang/oceanus_cost_model_training_data/ncf/ncf_large_adam_dense_g3.4.25.6_g3.4.25.7_g3.4.25.8_g3.4.25.9',
			# '/home/hao.zhang/oceanus_cost_model_training_data/ncf/ncf_large_adam_dense_g3.4.25.1_g3.4.25.2_g3.4.25.3_g3.4.25.4_3.4.25.6_g3.4.25.7_g3.4.25.8_g3.4.25.9',
			# '/home/hao.zhang/oceanus_cost_model_training_data/ncf/ncf_large_adam_dense_g3.4.25.1',
        ],
		'original_graph_item_path': '/home/christy.li/oceanus_cost_model_training_data/ncf/original_graph_item',
        'save_dir': os.path.join(expanduser('~'), 'oceanus_cost_model_training_data/ncf/predefined_checkpoints'),
		'save_prefix': 'ckpV1_ncf_large_adam_dense_orca_all',
		# 'save_prefix': 'ckpV2_ncf_large_adam_dense_orca',
		'baseline': 0.15,
		# 'baseline': 0.0,
		'scale': 0.5,
		'learning_rate': 0.01,
		'list_size': 2,
		'batch_size': 100,
		'ranking_loss_key': 'pairwise_logistic_loss',
		'model_version': 'v1',
		# 'model_version': 'v2',
		'do_train': False,
		'do_test': True,
		'checkpoint': '/home/christy.li/oceanus_cost_model_training_data/ncf/predefined_checkpoints/ckpV1_ncf_large_adam_dense_orca_all_600_0.83249_0.84517',
	},
	'bert': {
		'model_batch_size': 32,
		'model_seq_len': 128,
		'data_dir': [
			'/home/christy.li/oceanus_cost_model_training_data/bert/bert_3l_orca_16',
			# '/home/christy.li/oceanus_cost_model_training_data/bert/bert_6l_orca_15',
			# '/home/christy.li/oceanus_cost_model_training_data/bert/bert_12l_orca_15',
			# '/home/christy.li/oceanus_cost_model_training_data/bert/bert.12l_g4.4.50.1_g4.4.50.2',
			# '/home/christy.li/oceanus_cost_model_training_data/bert/bert.6l_g4.4.50.1_g4.4.50.2',
		],
        'original_graph_item_path': '/home/hao.zhang/oceanus_cost_model_training_data/bert/bert_original_graph_item_3l',
        'save_dir': '/home/christy.li/oceanus_cost_model_training_data/bert/predefined_checkpoints',
		'save_prefix': 'ckpV1_bert_orca',
		'baseline': 0.04,
		'scale': 0.5,
		'learning_rate': 0.01,
		'list_size': 2,
		'batch_size': 100,
		'ranking_loss_key': 'pairwise_logistic_loss',
		'do_train': False,
		'do_test': True,
		'model_version': 'v1',
		# 'model_version': 'v2',
		# 'checkpoint': '/home/christy.li/oceanus_cost_model_training_data/ncf/predefined_checkpoints/checkpoint_500',
		# 'checkpoint': '/home/christy.li/oceanus_cost_model_training_data/ncf/predefined_checkpoints/ckpV1_ncf_large_adam_dense_orca_16_300_0.90684_0.91947',
		# 'checkpoint': '/home/christy.li/oceanus_cost_model_training_data/ncf/predefined_checkpoints/ckpV1_ncf_large_adam_dense_orca_16_600_0.87000_0.71000',
		# 'checkpoint': '/home/christy.li/oceanus_cost_model_training_data/ncf/predefined_checkpoints/ckpV1_ncf_large_adam_dense_all_200_0.80568_0.81116',
		# 'checkpoint': '/home/christy.li/oceanus_cost_model_training_data/ncf/predefined_checkpoints/ckpV1_ncf_large_adam_dense_orca_200_0.81503_0.82009',
		# 'checkpoint': '/home/christy.li/oceanus_cost_model_training_data/ncf/predefined_checkpoints/ckpV2_ncf_large_adam_dense_orca_16_600_0.89737_0.92842',
		# 'checkpoint': '/home/christy.li/oceanus_cost_model_training_data/ncf/predefined_checkpoints/ckpV2_ncf_large_adam_dense_all_500_0.87666_0.85391',
		'checkpoint': '/home/christy.li/oceanus_cost_model_training_data/bert/predefined_checkpoints/ckpV1_bert_orca_400_0.93600_0.93889',
	},
	'resnet101': {
		'model_batch_size': 32,
		'model_seq_len': 1,
		'baseline': 0.5,
		'scale': 0.5,
		'data_dir': '',
		'learning_rate': 0.01,
		'list_size': 2,
		'batch_size': 100,
		'ranking_loss_key': 'pairwise_logistic_loss',
	},
}

def main(_):
	np.random.seed(110)

	# Hyperparameters
	# model_to_simulate = 'bert'
	model_to_simulate = 'ncf_large_adam_dense'
	data_dir = model_params[model_to_simulate]['data_dir']
	original_graph_item_path = model_params[model_to_simulate]['original_graph_item_path']
	batch_size = model_params[model_to_simulate]['batch_size']
	ranking_loss_key = model_params[model_to_simulate]['ranking_loss_key']
	learning_rate = model_params[model_to_simulate]['learning_rate']
	list_size = model_params[model_to_simulate]['list_size']
	baseline = model_params[model_to_simulate]['baseline']
	scale = model_params[model_to_simulate]['scale']
	save_dir = model_params[model_to_simulate]['save_dir']
	save_prefix = model_params[model_to_simulate]['save_prefix']
	do_train = model_params[model_to_simulate]['do_train']
	do_test = model_params[model_to_simulate]['do_test']
	checkpoint = model_params[model_to_simulate]['checkpoint']
	model_version = model_params[model_to_simulate]['model_version']

	# Create simulator
	simulator = PredefinedSimulator(original_graph_item_path,
	                                batch_size=model_params[model_to_simulate]['model_batch_size'],
	                                seq_len=model_params[model_to_simulate]['model_seq_len'])

	# Create features
	strategy_resource_files, Y = utils.laod_from_folders(data_dir)
	print("Createing features...")
	X = []
	with context.graph_mode():
		for strategy_file, resource_file in tqdm.tqdm(strategy_resource_files):
			x = simulator.create_features(Strategy.deserialize(strategy_file), ResourceSpec(resource_file))
			X.append(x)
	X = np.array(X, dtype=np.float)
	print("Finished createing features.")

	# Create model
	hidden_dim = 12
	W = tf.Variable(tf.random.uniform([hidden_dim, 1]), name='W', dtype=tf.float32)
	b = tf.Variable(0.0, name='b', dtype=tf.float32)
	if model_version == 'v2':
		W0 = tf.Variable(tf.random.uniform([hidden_dim, hidden_dim]), name='W0', dtype=tf.float32)
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
	def train_steps(inputs_iterator, total_steps):

		def train_step(input):
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
			l, a = train_step(inputs_iterator.get_next())
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
		train_set, valid_set, test_set = utils.split_dataset([X, Y], shuffle=True, train_ratio=0.7, test_ratio=0.15)
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
		EPOCHS = 2000
		eval_every_epochs = 100
		save_every_epochs = 100

		print("\nTrain model...")
		losses = []
		for epoch in range(EPOCHS):
			loss, acc = train_steps(inputs_iterator, total_train_steps)
			losses.extend(loss)
			avgloss = sum(losses) / float(len(losses))
			print('Step: {}, avgloss: {:.5f}'.format(epoch, avgloss))
			if (epoch+1) % eval_every_epochs == 0:
				print("\nEvaluate on valid set...")
				avg_valid_acc, *_= eval_steps(valid_iterator, total_valid_steps)
				print('avg_valid_acc: {}'.format(avg_valid_acc.numpy()))
				print("Evaluate on test set...")
				avg_test_acc, *_= eval_steps(test_iterator, total_test_steps)
				print('avg_test_acc: {}\n'.format(avg_test_acc.numpy()))
				print('W', W.numpy())
				print('b', b.numpy())

				if (epoch+1) % save_every_epochs == 0:
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


main(sys.argv)
