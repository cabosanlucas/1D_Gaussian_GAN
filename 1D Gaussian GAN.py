import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


class DataDistribution(object):
	"""docstring for DataDistribution"""
	def __init__(self, mu, sigma,range):
		self.mu = -mu
		self.sigma = sigma
		self.range = range
	def get_samples(self, N):
		samples = np.random.normal(size=N)
		samples.sort()
		return samples

	def get_labels(self, samples):
		labels=norm.pdf(samples,loc=self.mu,scale=self.sigma)
		return labels

	def plot(self, N):
		self.xs = np.linspace(-self.range,self.range,N)
		plt.plot(self.xs, self.get_samples(N))
		plt.show()


class GeneratorDistribution(object):
	"""docstring for GeneratorDistribution"""
	def __init__(self, range):
		self.range = range

	def sample(self, N):
		return np.linspace(-self.range, self.range, N) + np.random.random(N) * 0.01

def generator(input, hidden_size, copy=False):
	h0 = tf.nn.softplus(linear(input,hidden_size, "g0", copy))
	h1 = linear(h0,1,'g1', copy)		
	return h1

def discriminator(input, hidden_size, is_copy):
	h0 = tf.tanh(linear(input, hidden_size, 'd0', is_copy))
	h1 = tf.tanh(linear(h0,hidden_size,'d1', is_copy))
	h2 = mini_batch_layer(h1, copy = is_copy)
	h3 = tf.sigmoid(linear(h2, 1, 'd3', is_copy))
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
		if reuse:
			var_scope.reuse_variables()
		w = tf.get_variable('w', [input.get_shape()[1], output_dim], initializer = norm)
		b = tf.get_variable('b', [output_dim], initializer = const)
		return tf.matmul(input, w) +b 
		

z = tf.placeholder(tf.float32, shape=(None,1))
G = generator(z, 11, False)

x = tf.placeholder(tf.float32, shape=(None, 1))
D1 = discriminator(x, 11, False)

D2 = discriminator(G, 11, True)

cost_d = tf.reduce_mean(tf.log(D1) + tf.log(1-D2))
cost_g = tf.reduce_mean(tf.log(D2))

optimize_d = tf.train.GradientDescentOptimizer(0.01).minimize(1-cost_d)
optimize_g = tf.train.GradientDescentOptimizer(0.01).minimize(1-cost_g)

HIDDEN_LAYER_SIZE = 11
class GAN(object):

	"""docstring for GAN"""
	def __init__(self, data, gen, num_steps, mini_batch_size):
		self.data = data
		self.generator = gen
		self.num_steps = num_steps
		self.mini_batch_size = mini_batch_size

	def create_model(self):
		#pre-train discriminator
		with tf.variable_scope("D_pre"):
			self.pre_input = tf.placeholder(tf.float32, shape=(self.mini_batch_size, 1))
			self.pre_labels = tf.placeholder(tf.float32, shape = (self.mini_batch_size, 1))
			D_pre = discriminator(self.pre_input,HIDDEN_LAYER_SIZE, False)
			self.pre_cost_f = tf.reduce_mean(D_pre - self.pre_labels)
			self.pre_optimize = tf.train.GradientDescentOptimizer(0.01).minimize(pre_cost_f)


		with tf.variable_scope("Gen"):
			self.noise_input = tf.placeholder(tf.float32, shape = (self.mini_batch_size, 1))
			self.generated_data = generator(self.noise_input, HIDDEN_LAYER_SIZE)

		with tf.variable_scope("Disc") as scope:
			self.real_data = tf.placeholder(tf.float32, shape=(None, 1))
			self.D1 = discriminator(self.real_data, 11, False)
			self.scope.reuse_variables()
			self.D2 = discriminator(self.generated_data, 11, True)

		cost_d = tf.reduce_mean(tf.log(self.D1) + tf.log(1-D2))
		cost_g = tf.reduce_mean(tf.log(self.D2))

		optimize_d = tf.train.GradientDescentOptimizer(0.01).minimize(1-cost_d)
		optimize_g = tf.train.GradientDescentOptimizer(0.01).minimize(1-cost_g)

	def train(self):
		with tf.Session() as session:
			tf.global_variables_initializer().run()

			#pretraining discrimnator
			data_dist = DataDistribution()
			PRETRAIN_STEPS = 1000
			for step in range(PRETRAIN_STEPS):
				pre_data = (np.random.random(self.mini_batch_size) - 0.5) * 10
				labels = norm.pdf(pre_data, loc = self.data.mu, scale = self.data.sigma)
				pretrain_cost, _ = session.run([self.pre_cost_f, self.pre_optimize], {
					self.pre_input : np.reshape(pre_data, (self.mini_batch_size, 1)),
					self.pre_labels : np.reshape(labels, (self.mini_batch_size, 1))
					})



#for i in range(10000):
data_dist = DataDistribution(-1,1,5)
data = data_dist.get_samples(1000)
data_dist.plot(10000)

