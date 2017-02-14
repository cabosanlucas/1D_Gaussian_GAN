import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


class DataDistribution(object):
	"""docstring for DataDistribution"""
	def __init__(self, mu, sigma):
		self.mu = mu
		self.sigma = sigma
		self.range = range

	def get_samples(self, N):
		samples = np.random.normal(self.mu, self.sigma, N)
		samples.sort()
		return samples

	def plot(self, N):
		plt.plot(self.get_samples(N))
		plt.show()


class GeneratorDistribution(object):
	"""docstring for GeneratorDistribution"""
	def __init__(self, range):
		self.range = range

	def get_samples(self, N):
		return np.linspace(-self.range, self.range, N) + np.random.random(N) * 0.01

	def plot(self, N):
		self.xs = np.linspace(-self.range,self.range,N)
		plt.plot(self.xs, self.get_samples(N))
		plt.show()

def generator(input, hidden_size, copy=False):
	h0 = tf.nn.softplus(linear(input,hidden_size, "g0", copy))
	h1 = linear(h0,1,'g1', copy)		
	return h1

def discriminator(input, hidden_size, is_copy):
	h0 = tf.tanh(linear(input, hidden_size, 'd0', is_copy))
	print hidden_size
	h1 = tf.tanh(linear(h0,hidden_size,'d1', is_copy))
	h2 = mini_batch_layer(h1, copy = is_copy)
	h3 = tf.sigmoid(linear(h2, 1, 'd3', is_copy), name = "h3")
	return h3

def mini_batch_layer(input, num_kernels=5, kernel_dim=3, copy=False):
	x = linear(input, num_kernels * kernel_dim, scope='minibatch', stddev=0.02, reuse = copy)
	activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
	diffs = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
	abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
	minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
	return tf.concat(1, [input, minibatch_features])


def linear(input, output_dim, scope, reuse=False, stddev=1.0):
	norm = tf.random_normal_initializer(stddev=stddev)
	const = tf.constant_initializer(0.0)
	with tf.variable_scope(scope) as var_scope:
		if False:
			var_scope.reuse_variables()
		w = tf.get_variable('w', [input.get_shape()[1], output_dim], initializer = norm)
		b = tf.get_variable('b', [output_dim], initializer = const)
		return tf.matmul(input, w) +b 

HIDDEN_LAYER_SIZE = 11
class GAN(object):

	"""docstring for GAN"""
	def __init__(self, data, gen, num_steps, mini_batch_size):
		self.data = data
		self.generator = gen
		self.num_steps = num_steps
		self.mini_batch_size = mini_batch_size
		self.create_model()

	def create_model(self):
		#pre-train discriminator
		with tf.variable_scope("D_pre"):
			self.pre_input = tf.placeholder(tf.float32, shape=(self.mini_batch_size, 1))
			self.pre_labels = tf.placeholder(tf.float32, shape = (self.mini_batch_size, 1))
			D_pre = discriminator(self.pre_input,HIDDEN_LAYER_SIZE, False)
			self.pre_cost_f = tf.reduce_mean(D_pre - self.pre_labels)
			self.pre_optimize = tf.train.GradientDescentOptimizer(0.001).minimize(self.pre_cost_f)


		with tf.variable_scope("Gen"):
			self.noise_input = tf.placeholder(tf.float32, shape = (self.mini_batch_size, 1))
			self.generated_data = generator(self.noise_input, 11)

		with tf.variable_scope("Disc") as scope:
			self.real_data = tf.placeholder(tf.float32, shape=(self.mini_batch_size, 1))
			print tf.Print(self.real_data, [self.real_data])
			self.D1 = discriminator(self.real_data, HIDDEN_LAYER_SIZE, False)
			scope.reuse_variables()
			self.D2 = discriminator(self.generated_data, HIDDEN_LAYER_SIZE, True)

		self.cost_d = tf.reduce_mean(-tf.log(self.D1) - tf.log(1-self.D2))
		self.cost_g = tf.reduce_mean(-tf.log(self.D2))
		
		self.d_pre_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="D_pre")
		self.d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Disc")
		self.g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Gen")

		self.optimize_d = tf.train.GradientDescentOptimizer(0.005).minimize(self.cost_d)
		self.optimize_g = tf.train.GradientDescentOptimizer(0.005).minimize(self.cost_g)

	def train(self):
		with tf.Session() as session:
			tf.global_variables_initializer().run()

			#pretrain discrimnator
			PRETRAIN_STEPS = 1500
			for step in range(PRETRAIN_STEPS):
				pre_data = (np.random.random(self.mini_batch_size) - 0.5) * 10
				labels = norm.pdf(pre_data, loc = self.data.mu, scale = self.data.sigma)
				pretrain_cost, _ = session.run([self.pre_cost_f, self.pre_optimize], {
					self.pre_input : np.reshape(pre_data, (self.mini_batch_size, 1)),
					self.pre_labels : np.reshape(labels, (self.mini_batch_size, 1))
					})
			#save pre-traning weights
			self.weightsD = session.run(self.d_pre_params)
			#copy pre-training weights
			for i, v in enumerate(self.d_params):
				session.run(v.assign(self.weightsD[i]))
			print tf.Print(self.cost_d, [self.cost_d])
			print self.weightsD

			for step in xrange(self.num_steps):
				#update discriminator
				real_data = self.data.get_samples(self.mini_batch_size)
				gen_data = self.generator.get_samples(self.mini_batch_size)
				loc_d_cost, _ = session.run([self.cost_d, self.optimize_d], {
						self.real_data: np.reshape(real_data, (self.mini_batch_size, 1)),
						self.noise_input: np.reshape(gen_data, (self.mini_batch_size,1)) 
					})

				#update generator
				noise = self.generator.get_samples(self.mini_batch_size)
				loc_g_cost, _ = session.run([self.cost_g, self.optimize_g], {
					self.noise_input: np.reshape(noise, (self.mini_batch_size, 1))
				})

				if(step % 10 == 0):
					print('{}: {}\t{}'.format(step, loc_d_cost, loc_g_cost))
			self._plot_distributions(session)

	def _samples(self, session, num_points=10000, num_bins=100):
		'''
		Return a tuple (db, pd, pg), where db is the current decision
		boundary, pd is a histogram of samples from the data distribution,
		and pg is a histogram of generated samples.
		'''
		xs = np.linspace(-self.generator.range, self.generator.range, num_points)
		bins = np.linspace(-self.generator.range, self.generator.range, num_bins)

		# decision boundary
		db = np.zeros((num_points, 1))
		for i in range(num_points // self.mini_batch_size):
			db[self.mini_batch_size * i:self.mini_batch_size * (i + 1)] = session.run(self.D1, {
				self.real_data: np.reshape(
					xs[self.mini_batch_size * i:self.mini_batch_size * (i + 1)],
					(self.mini_batch_size, 1)
				)
			})

		# data distribution
		d = self.data.get_samples(num_points)
		pd, _ = np.histogram(d, bins=bins, density=True)

		# generated samples
		zs = np.linspace(-self.generator.range, self.generator.range, num_points)
		g = np.zeros((num_points, 1))
		for i in range(num_points // self.mini_batch_size):
			g[self.mini_batch_size * i:self.mini_batch_size * (i + 1)] = session.run(self.generated_data, {
				self.noise_input: np.reshape(
					zs[self.mini_batch_size * i:self.mini_batch_size * (i + 1)],
					(self.mini_batch_size, 1)
				)
			})
		pg, _ = np.histogram(g, bins=bins, density=True)

		return db, pd, pg

	def _plot_distributions(self, session):
		db, pd, pg = self._samples(session)
		db_x = np.linspace(-self.generator.range, self.generator.range, len(db))
		p_x = np.linspace(-self.generator.range, self.generator.range, len(pd))
		f, ax = plt.subplots(1)
		ax.plot(db_x, db, label='decision boundary')
		ax.set_ylim(0, 1)
		plt.plot(p_x, pd, label='real data')
		plt.plot(p_x, pg, label='generated data')
		plt.title('1D Generative Adversarial Network')
		plt.xlabel('Data values')
		plt.ylabel('Probability density')
		plt.legend()
		plt.show()
		
model = GAN(DataDistribution(1.0, 0.5), GeneratorDistribution(5), 1500, 12)
model.train()