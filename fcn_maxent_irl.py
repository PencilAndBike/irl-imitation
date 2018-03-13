import numpy as np
import tensorflow as tf
import mdp.gridworld as gridworld
import mdp.value_iteration as value_iteration
import img_utils
import tf_utils
from utils import *

class FCNIRL:
  def __init__(self, input_shape, out_shape, lr, n_h1=400, n_h2=300, l2=10, name='deep_irl_fc', gpu_fraction=0.2):
    # self.n_input = n_input
    self.input_shape = input_shape
    self.out_shape = out_shape
    self.lr = lr
    self.n_h1 = n_h1
    self.n_h2 = n_h2
    self.name = name
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    self.input_s, self.reward, self.theta = self._build_network(self.name)
    self.optimizer = tf.train.GradientDescentOptimizer(lr)
    
    self.grad_r = tf.placeholder(tf.float32, [None]+list(out_shape)+[1])
    self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.theta])
    self.grad_l2 = tf.gradients(self.l2_loss, self.theta)
    
    print "reward shape: ", self.reward.shape
    print "grad shape: ", self.grad_r.shape
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
      # reward = tf_utils.conv2d(input_s, 1, [1,1])
      # reward = tf_utils.conv2d(conv1, 1, [1,1])
      conv = tf_utils.conv2d(input_s, 32, [4,4], stride=4)
      conv = tf_utils.conv2d(conv, 32, [4,4], stride=4)
      conv = tf_utils.conv2d(conv, 16, [2,2], stride=2)
      conv = tf_utils.conv2d(conv, 16, [2,2])
      reward = tf_utils.conv2d(conv, 1, [1,1])
      # reward = tf_utils.conv2d(conv1, 1, [1,1])
      # conv3 = tf_utils.conv2d(conv2, 1, [1,1])
      # conv4 = tf.concat([input_s, conv3], axis=-1)
      # reward = tf_utils.conv2d(conv4, 1, [1,1])
      # conv2 = tf_utils.conv2d(conv1, 4, [2,2])
      # reward = tf_utils.conv2d(conv1, 1, [2,2])
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
  
  for s in range(N_STATES):
    for t in range(T - 1):
      if deterministic:
        mu[s, t + 1] = sum([mu[pre_s, t] * P_a[pre_s, s, int(policy[pre_s])] for pre_s in range(N_STATES)])
      else:
        mu[s, t + 1] = sum(
          [sum([mu[pre_s, t] * P_a[pre_s, s, a1] * policy[pre_s, a1] for a1 in range(N_ACTIONS)]) for pre_s in
           range(N_STATES)])
  
  # for t in range(T - 1):
  #   for s in range(N_STATES):
  #     if deterministic:
  #       mu[s, t + 1] = sum([mu[pre_s, t] * P_a[pre_s, s, int(policy[pre_s])] for pre_s in range(N_STATES)])
  #     else:
  #       mu[s, t + 1] = sum(
  #         [sum([mu[pre_s, t] * P_a[pre_s, s, a1] * policy[pre_s, a1] for a1 in range(N_ACTIONS)]) for pre_s in
  #          range(N_STATES)])
  
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


def demo_sparse_svf(traj, n_states):
  # def idx2pos(idx):
  #   return (idx % h, idx / h)
  #
  # traj_pos = []
  # dis = []
  # for s, a, ns, r, is_done in traj:
  #   traj_pos.append(ns)
  #
  # for idx in range(n_states):
  #   pos = idx2pos(idx)
  #   for pos_on_traj in traj_pos:
  #     dis.append(np.linalg.norm(np.array(pos)-np.array(pos_on_traj)))
  p = np.zeros(n_states)
  l = len(traj)
  for i, step in enumerate(traj):
    p[step.next_state] = (i+1.0)/l
  return p
  
  
def fcn_maxent_irl(feat_maps, out_shape, P_as, gamma, trajs, lr, n_iters, gpu_fraction):
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
  
  N_TRAJ, N_STATES, _, N_ACTIONS = np.shape(P_as)
  
  # init nn model
  nn_r = FCNIRL(feat_maps.shape[1:], out_shape, lr, 3, 3, gpu_fraction)
  
  for itr in range(n_iters):
    grad_rs = []
    print "======itr {}=======".format(itr)
    for i in range(N_TRAJ):
      print "VI traj {}".format(i)
      feat_map, P_a, traj = feat_maps[i], P_as[i], trajs[i]
      feat_map = np.array([feat_map])
      mu_D = demo_sparse_svf(traj, N_STATES)
      rewards = nn_r.get_rewards(feat_map)
      rewards = np.reshape(rewards, N_STATES, order='F')
      # rewards = np.reshape(rewards, feat_map.shape[1]*feat_map.shape[2], order='F')
      _, policy = value_iteration.value_iteration(P_a, rewards, gamma, error=0.01, deterministic=True)
      mu_exp = compute_state_visition_freq(P_a, gamma, [traj], policy, deterministic=True)
      grad_r = mu_D - mu_exp
      grad_r = np.reshape(grad_r, (out_shape[0], out_shape[1], 1), order='F')
      grad_rs.append(grad_r)
    grad_rs = np.array(grad_rs)
    grad_theta, l2_loss, grad_norm = nn_r.apply_grads(feat_maps, grad_rs)
    
  rewards = nn_r.get_rewards(feat_maps)
  n_rewards = []
  # for reward, feat_map in zip(rewards, feat_maps):
  #   n_rewards.append(np.reshape(reward, feat_map.shape[0]*feat_map.shape[1], order='F'))
  for reward in rewards:
    n_rewards.append(np.reshape(reward, out_shape[0]*out_shape[1], order='F'))
  # rewards = np.reshape(rewards, feat_map.shape[0]*feat_map.shape[1], order='F')
  # norm_rewards = []
  # for reward, feat_map in zip(rewards, feat_maps):
  #   reward = np.reshape(reward, feat_map.shape[0] * feat_map.shape[1], order='F')
  #   norm_reward = normalize(reward)
  #   norm_rewards.append(norm_reward)
  # return norm_rewards
  return n_rewards
