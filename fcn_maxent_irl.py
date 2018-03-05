import numpy as np
import tensorflow as tf
import mdp.gridworld as gridworld
import mdp.value_iteration as value_iteration
import img_utils
import tf_utils
from utils import *


class FCNIRL:
  def __init__(self, input_shape, lr, n_h1=400, n_h2=300, l2=10, name='deep_irl_fc', gpu_fraction=0.2):
    # self.n_input = n_input
    self.input_shape = input_shape
    self.lr = lr
    self.n_h1 = n_h1
    self.n_h2 = n_h2
    self.name = name
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    self.input_s, self.reward, self.theta = self._build_network(self.name)
    self.optimizer = tf.train.GradientDescentOptimizer(lr)
    
    self.grad_r = tf.placeholder(tf.float32, [None]+list(input_shape[:2])+[1])
    self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.theta])
    self.grad_l2 = tf.gradients(self.l2_loss, self.theta)
    
    self.grad_theta = tf.gradients(self.reward, self.theta, -self.grad_r)
    # apply l2 loss gradients
    self.grad_theta = [tf.add(l2 * self.grad_l2[i], self.grad_theta[i]) for i in range(len(self.grad_l2))]
    self.grad_theta, _ = tf.clip_by_global_norm(self.grad_theta, 100.0)
    
    self.grad_norms = tf.global_norm(self.grad_theta)
    self.optimize = self.optimizer.apply_gradients(zip(self.grad_theta, self.theta))
    self.sess.run(tf.global_variables_initializer())
  
  def _build_network(self, name):
    input_s = tf.placeholder(tf.float32, [None]+list(self.input_shape))
    # input_s = tf.placeholder(tf.float32, [None]+self.input_shape+[1])
    # with tf.variable_scope(name):
    #   fc1 = tf_utils.fc(input_s, self.n_h1, scope="fc1", activation_fn=tf.nn.elu,
    #                     initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
    #   fc2 = tf_utils.fc(fc1, self.n_h2, scope="fc2", activation_fn=tf.nn.elu,
    #                     initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
    #   reward = tf_utils.fc(fc2, 1, scope="reward")
    with tf.variable_scope(name):
      print "input_s shape: ", input_s.shape
      reward = tf_utils.conv2d(input_s, 1, [1,1])
      # reward = tf_utils.conv2d(conv1, 1, [1,1])
      # conv2 = tf_utils.conv2d(conv1, 16, [4,4])
      # reward = tf_utils.conv2d(conv2, 1, [4,4])
    theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
    return input_s, reward, theta
  
  def get_theta(self):
    return self.sess.run(self.theta)
  
  def get_rewards(self, states):
    rewards = self.sess.run(self.reward, feed_dict={self.input_s: states})
    return rewards
  
  def apply_grads(self, feat_map, grad_r):
    # grad_r = np.reshape(grad_r, [-1, 1])
    # feat_map = np.reshape(feat_map, [-1, self.n_input])
    _, grad_theta, l2_loss, grad_norms = self.sess.run([self.optimize, self.grad_theta, self.l2_loss, self.grad_norms],
                                                       feed_dict={self.grad_r: grad_r, self.input_s: feat_map})
    return grad_theta, l2_loss, grad_norms


def compute_state_visition_freq(P_a, gamma, trajs, policy, deterministic=True):
  """compute the expected states visition frequency p(s| theta, T)
  using dynamic programming

  inputs:
    P_a     NxNxN_ACTIONS matrix - transition dynamics
    gamma   float - discount factor
    trajs   list of list of Steps - collected from expert
    policy  Nx1 vector (or NxN_ACTIONS if deterministic=False) - policy


  returns:
    p       Nx1 vector - state visitation frequencies
  """
  N_STATES, _, N_ACTIONS = np.shape(P_a)
  
  T = len(trajs[0])
  # mu[s, t] is the prob of visiting state s at time t
  mu = np.zeros([N_STATES, T])
  
  for traj in trajs:
    mu[traj[0].cur_state, 0] += 1
  mu[:, 0] = mu[:, 0] / len(trajs)
  
  # for s in range(N_STATES):
  #   for t in range(T - 1):
  #     if deterministic:
  #       mu[s, t + 1] = sum([mu[pre_s, t] * P_a[pre_s, s, int(policy[pre_s])] for pre_s in range(N_STATES)])
  #     else:
  #       mu[s, t + 1] = sum(
  #         [sum([mu[pre_s, t] * P_a[pre_s, s, a1] * policy[pre_s, a1] for a1 in range(N_ACTIONS)]) for pre_s in
  #          range(N_STATES)])
  
  for t in range(T - 1):
    for s in range(N_STATES):
      if deterministic:
        mu[s, t + 1] = sum([mu[pre_s, t] * P_a[pre_s, s, int(policy[pre_s])] for pre_s in range(N_STATES)])
      else:
        mu[s, t + 1] = sum(
          [sum([mu[pre_s, t] * P_a[pre_s, s, a1] * policy[pre_s, a1] for a1 in range(N_ACTIONS)]) for pre_s in
           range(N_STATES)])
  
  p = np.sum(mu, 1)
  return p


def demo_svf(trajs, n_states):
  """
  compute state visitation frequences from demonstrations

  input:
    trajs   list of list of Steps - collected from expert
  returns:
    p       Nx1 vector - state visitation frequences
  """
  
  p = np.zeros(n_states)
  for traj in trajs:
    for step in traj:
      p[step.cur_state] += 1
  p = p / len(trajs)
  return p


def fcn_maxent_irl(feat_map, P_a, gamma, trajs, lr, n_iters, gpu_fraction):
  """
  Maximum Entropy Inverse Reinforcement Learning (Maxent IRL)

  inputs:
    feat_map    1xDxDx1 matrix - the features of states map
    P_a         NxNxN_ACTIONS matrix - P_a[s0, s1, a] is the transition prob of
                                       landing at state s1 when taking action
                                       a at state s0
    gamma       float - RL discount factor
    trajs       a list of demonstrations
    lr          float - learning rate
    n_iters     int - number of optimization steps

  returns
    rewards     Nx1 vector - recoverred state rewards
  """
  
  # tf.set_random_seed(1)
  
  N_STATES, _, N_ACTIONS = np.shape(P_a)
  
  # init nn model
  nn_r = FCNIRL(feat_map.shape[1:], lr, 3, 3, gpu_fraction)
  
  # find state visitation frequencies using demonstrations
  mu_D = demo_svf(trajs, N_STATES)
  print "mu_D: ",  mu_D.reshape((feat_map.shape[1]*feat_map.shape[2], ))
  
  # training
  for iteration in range(n_iters):
    if iteration % (n_iters / 10) == 0:
      print 'iteration: {}'.format(iteration)
    
    # compute the reward matrix
    rewards = nn_r.get_rewards(feat_map)
    # rewards = np.reshape(rewards, [-1, 1])
    # print "rewards: ", rewards
    # print "rewards shape: ", rewards.shape
    # rewards = rewards[0][:,:,0]
    rewards = np.reshape(rewards, feat_map.shape[1]*feat_map.shape[2], order='F')
    # print "reshaped rewards: ", rewards
    # compute policy
    _, policy = value_iteration.value_iteration(P_a, rewards, gamma, error=0.01, deterministic=True)
    
    # compute expected svf
    mu_exp = compute_state_visition_freq(P_a, gamma, trajs, policy, deterministic=True)
    print "mu_exp: ", mu_exp.reshape((feat_map.shape[1]*feat_map.shape[2], ))
    
    # compute gradients on rewards:
    grad_r = mu_D - mu_exp
    # grad_r = np.repeat(grad_r, feat_map.shape[0]*feat_map.shape[1]).reshape((-1, feat_map.shape[0], feat_map.shape[1], 1))
    # grad_r = grad_r.reshape((1, feat_map.shape[1], feat_map.shape[2], 1))
    # print "grad_r: ", grad_r
    grad_r = np.reshape(grad_r, (1, feat_map.shape[1], feat_map.shape[2], 1), order='F')
    # grad_r = np.reshape(grad_r, (1, feat_map.shape[1], feat_map.shape[2], 1))
    # print "grad_r shape: ", grad_r.shape
    # print "reshaped grad_r: ", grad_r
    # apply gradients to the neural network
    grad_theta, l2_loss, grad_norm = nn_r.apply_grads(feat_map, grad_r)
  
  rewards = nn_r.get_rewards(feat_map)
  
  # rewards = np.zeros(feat_map.shape)
  # rewards[0][feat_map.shape[1]-1][feat_map.shape[2]-1][0] = \
  #   rewards[0][0][feat_map.shape[2]-1][0] = \
  #   rewards[0][feat_map.shape[1]-1][0][0] = 1.0
  
  rewards = np.reshape(rewards, feat_map.shape[1] * feat_map.shape[2], order='F')
  print "rewards: ", rewards
  center_rewards = rewards - rewards.mean()
  print "rewards reduce mean: ", center_rewards
  # print "gaussian normalize rewards: ", rewards - rewards.mean()
  norm_rewards = normalize(rewards)
  print "normalize rewards: ", norm_rewards
  # norm2_rewards = (rewards - rewards.mean()) / rewards.std()
  # print "normalize2 rewards: ", norm2_rewards
  # sigmoid_center_rewards = 1 / (1 + np.exp(-center_rewards))
  # print "sigmoid rewards: ", sigmoid_center_rewards
  # sigmoid_norm2_rewards = 1 / (1 + np.exp(-norm2_rewards))
  # print "sigmoid norm2 rewards: ", sigmoid_norm2_rewards
  # return sigmoid(normalize(rewards))
  # return normalize(rewards)
  # return sigmoid_norm2_rewards
  # return sigmoid_center_rewards
  return norm_rewards



