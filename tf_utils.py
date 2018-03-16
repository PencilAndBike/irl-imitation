"""Utility functions for tensorflow"""
import tensorflow as tf
import numpy as np


def max_pool(x, k_sz=[2, 2]):
  """max pooling layer wrapper
  Args
    x:      4d tensor [batch, height, width, channels]
    k_sz:   The size of the window for each dimension of the input tensor
  Returns
    a max pooling layer
  """
  return tf.nn.max_pool(
      x, ksize=[
          1, k_sz[0], k_sz[1], 1], strides=[
          1, k_sz[0], k_sz[1], 1], padding='SAME')


def conv2d(x, n_kernel, k_sz, stride=1, name='conv'):
  """convolutional layer with relu activation wrapper
  Args:
    x:          4d tensor [batch, height, width, channels]
    n_kernel:   number of kernels (output size)
    k_sz:       2d array, kernel size. e.g. [8,8]
    stride:     stride
  Returns
    a conv2d layer
  """
  with tf.variable_scope(name):
    W = tf.Variable(tf.random_normal([k_sz[0], k_sz[1], int(x.get_shape()[3]), n_kernel]))
    b = tf.Variable(tf.random_normal([n_kernel]))
    
    # W = tf.Variable(tf.ones([k_sz[0], k_sz[1], int(x.get_shape()[3]), n_kernel]))
    # b = tf.Variable(tf.zeros([n_kernel]))
    # - strides[0] and strides[1] must be 1
    # - padding can be 'VALID'(without padding) or 'SAME'(zero padding)
    #     - http://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t
    conv = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
    conv = tf.nn.bias_add(conv, b)  # add bias term
    # rectified linear unit: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    # return tf.nn.relu(conv)
    return conv


def fc(x, n_output, activation_fn=None, initializer=None, name="fc"):
  """fully connected layer with relu activation wrapper
  Args
    x:          2d tensor [batch, n_input]
    n_output    output size
  """
  with tf.variable_scope(name):
    if initializer is None:
      # default initialization
      W = tf.Variable(tf.random_normal([int(x.get_shape()[1]), n_output]))
      b = tf.Variable(tf.random_normal([n_output]))
    else:
      W = tf.get_variable("W", shape=[int(x.get_shape()[1]), n_output], initializer=initializer)
      b = tf.get_variable("b", shape=[n_output],
                          initializer=tf.constant_initializer(.0, dtype=tf.float32))
    fc1 = tf.add(tf.matmul(x, W), b)
    if not activation_fn is None:
      fc1 = activation_fn(fc1)
  return fc1


def bn(x, is_train, name='bn'):
  moving_average_decay = 0.9
  # moving_average_decay = 0.99
  # moving_average_decay_init = 0.99
  with tf.variable_scope(name):
    decay = moving_average_decay
    # if global_step is None:
    # decay = moving_average_decay
    # else:
    # decay = tf.cond(tf.greater(global_step, 100)
    # , lambda: tf.constant(moving_average_decay, tf.float32)
    # , lambda: tf.constant(moving_average_decay_init, tf.float32))
    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
    # with tf.device('/CPU:0'):
    mu = tf.get_variable('mu', batch_mean.get_shape(), tf.float32,
                         initializer=tf.zeros_initializer(), trainable=False)
    sigma = tf.get_variable('sigma', batch_var.get_shape(), tf.float32,
                            initializer=tf.ones_initializer(), trainable=False)
    beta = tf.get_variable('beta', batch_mean.get_shape(), tf.float32,
                           initializer=tf.zeros_initializer())
    gamma = tf.get_variable('gamma', batch_var.get_shape(), tf.float32,
                            initializer=tf.ones_initializer())
    # BN when training
    update = 1.0 - decay
    # with tf.control_dependencies([tf.Print(decay, [decay])]):
    # update_mu = mu.assign_sub(update*(mu - batch_mean))
    update_mu = mu.assign_sub(update * (mu - batch_mean))
    update_sigma = sigma.assign_sub(update * (sigma - batch_var))
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mu)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_sigma)
  
    mean, var = tf.cond(is_train, lambda: (batch_mean, batch_var),
                        lambda: (mu, sigma))
    bn = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)
  
    # bn = tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, 1e-5)
  
    # bn = tf.contrib.layers.batch_norm(inputs=x, decay=decay,
    # updates_collections=[tf.GraphKeys.UPDATE_OPS], center=True,
    # scale=True, epsilon=1e-5, is_training=is_train,
    # trainable=True)
  return bn


def flatten(x):
  """flatten a 4d tensor into 2d
  Args
    x:          4d tensor [batch, height, width, channels]
  Returns a flattened 2d tensor
  """
  return tf.reshape(x, [-1, int(x.get_shape()[1] * x.get_shape()[2] * x.get_shape()[3])])


def update_target_graph(from_scope, to_scope):
  from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
  to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

  op_holder = []
  for from_var, to_var in zip(from_vars, to_vars):
      op_holder.append(to_var.assign(from_var))
  return op_holder


# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
  def _initializer(shape, dtype=None, partition_info=None):
      out = np.random.randn(*shape).astype(np.float32)
      out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
      return tf.constant(out)
  return _initializer
