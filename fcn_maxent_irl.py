import numpy as np
import tensorflow as tf
import mdp.gridworld as gridworld
import mdp.value_iteration as value_iteration
import img_utils
import tf_utils
from utils import *
import time
from multiprocessing import Pool, Manager

class FCNIRL:
  def __init__(self, input_shape, out_shape, lr, l2=10, name='deep_irl_fc', gpu_fraction=0.2,
               is_train=True):
    # self.n_input = n_input
    self.input_shape = input_shape
    self.out_shape = out_shape
    self.lr = lr
    # self.lr = tf.placeholder(tf.float32, name="lr")
    self.name = name
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    self.is_train = tf.placeholder(tf.bool, name="is_train")
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
  
  # def get_lr(self, step, lr_decay=0.1, lr_decay_steps=2000):
  
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
      conv1 = tf_utils.conv2d(input_s, 32, [4,4], stride=2, name='conv1')
      # conv1 = tf_utils.max_pool(conv1)
      conv1 = tf_utils.bn(conv1, is_train=self.is_train, name='bn1')
      conv2 = tf_utils.conv2d(conv1, 32, [4,4], stride=2, name='conv2')
      # conv2 = tf_utils.max_pool(conv2)
      conv2 = tf_utils.bn(conv2, is_train=self.is_train, name='bn2')
      conv3 = tf_utils.conv2d(conv2, 32, [2,2], stride=1, name='conv3')
      conv3 = tf_utils.bn(conv3, is_train=self.is_train, name='bn3')
      conv4 = tf_utils.conv2d(conv3, 64, [2,2], name='conv4')
      conv4 = tf_utils.bn(conv4, is_train=self.is_train, name='bn4')
      reward = tf_utils.conv2d(conv4, 1, [1,1], name='reward')
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
  
  def get_rewards(self, states, is_train=True):
    rewards = self.sess.run(self.reward, feed_dict={self.input_s: states, self.is_train: is_train})
    return rewards
  
  def apply_grads(self, feat_map, grad_r, is_train=True):
    # grad_r = np.reshape(grad_r, [-1, 1])
    # feat_map = np.reshape(feat_map, [-1, self.n_input])
    _, grad_theta, l2_loss, grad_norms = self.sess.run([self.optimize, self.grad_theta, self.l2_loss, self.grad_norms],
                                                       feed_dict={self.grad_r: grad_r, self.input_s: feat_map,
                                                                  self.is_train: is_train})
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
  mu = np.zeros([N_STATES, T+1])
  
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
  
  policy_hot = np.zeros((N_STATES, N_ACTIONS))
  for i in range(len(policy)):
    policy_hot[i, policy[i]] = 1

  P_a_T = P_a.transpose(0, 2, 1)
  SS = []
  for i in range(N_STATES):
    SS.append(np.matmul(policy_hot[i], P_a_T[i]))
  SS = np.array(SS)
  for t in range(T):
    mu[:, t+1] = np.matmul(mu[:, t], SS)

  # for t in range(T):
  #   for s in range(N_STATES):
  #     if deterministic:
  #       mu[s, t + 1] = sum([mu[pre_s, t] * P_a[pre_s, s, int(policy[pre_s])] for pre_s in range(N_STATES)])
  #     else:
  #       mu[s, t + 1] = sum(
  #         [sum([mu[pre_s, t] * P_a[pre_s, s, a1] * policy[pre_s, a1] for a1 in range(N_ACTIONS)]) for pre_s in
  #          range(N_STATES)])
  
  p = np.sum(mu, 1)
  return p


def demo_svf(traj, n_states):
  """
  compute state visitation frequences from demonstrations

  input:
    trajs   list of list of Steps - collected from expert
  returns:
    p       Nx1 vector - state visitation frequences
  """
  
  p = np.zeros(n_states)
  # for traj in trajs:
  #   for step in traj:
  #     p[step.cur_state] += 1
  # p = p / len(trajs)
  
  for step in traj:
    # p[step.next_state] += 1.0
    p[step.cur_state] += 1.0
  
  p[step.next_state] += 1.0
   
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


def fcn_maxent_irl(inputs, nn_r, P_a, gamma, t_trajs, lr, n_iters, gpu_fraction, ckpt_path, batch_size=16,
                   max_itr=100):
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
  
  N_STATES, _, N_ACTIONS = np.shape(P_a)
  out_shape = nn_r.out_shape

  def get_grad(feat_map, traj):
    feat_map = np.array([feat_map])
    # mu_D = demo_sparse_svf(traj, N_STATES)
    mu_D = demo_svf(traj, N_STATES)
    # print "mu_D:\n", mu_D
    reward = nn_r.get_rewards(feat_map, is_train=True)
    # print "rewards\n", rewards
    reward = np.reshape(reward, N_STATES, order='F')
    reward = normalize(reward)
    # rewards = np.reshape(rewards, feat_map.shape[1]*feat_map.shape[2], order='F')
    _, policy = value_iteration.value_iteration(P_a, reward, gamma, error=0.1, deterministic=True, max_itrs=max_itr)
    mu_exp = compute_state_visition_freq(P_a, gamma, [traj], policy, deterministic=True)
    # print "mu_exp:\n", mu_exp
    grad_r = mu_D - mu_exp
    grad_r = np.reshape(grad_r, (out_shape[0], out_shape[1], 1), order='F')
    # print 'grad_r {}: \n {}'.format(i, grad_r)
    return grad_r
    # grad_rs[i] = grad_r

  # init nn model
  saver = tf.train.Saver(tf.global_variables())
  for itr in range(n_iters):
    t = time.time()
    print "--------itr {}--------".format(itr)
    ids = np.random.randint(len(inputs), size=batch_size)
    feat_maps = inputs[ids]
    trajs = [t_trajs[i] for i in ids]
    print 'batch_size=', batch_size
    # manager = Manager()
    # grad_rs = manager.list([None]*batch_size)
    grad_rs = []
    # p = Pool(batch_size)
    
    for i in range(batch_size):
      feat_map, traj = feat_maps[i], trajs[i]
      grad_rs.append(get_grad(feat_map, traj))
      # p.apply_async(get_grad, args=(grad_rs, i, feat_map, traj))
    # p.close()
    # p.join()
    # print "join"
      # feat_map = np.array([feat_map])
      # # mu_D = demo_sparse_svf(traj, N_STATES)
      # mu_D = demo_svf(traj, N_STATES)
      # # print "mu_D:\n", mu_D
      # reward = nn_r.get_rewards(feat_map, is_train=True)
      # # print "rewards\n", rewards
      # reward = np.reshape(reward, N_STATES, order='F')
      # reward = normalize(reward)
      # # rewards = np.reshape(rewards, feat_map.shape[1]*feat_map.shape[2], order='F')
      # _, policy = value_iteration.value_iteration(P_a, reward, gamma, error=0.1, deterministic=True, max_itrs=max_itr)
      # mu_exp = compute_state_visition_freq(P_a, gamma, [traj], policy, deterministic=True)
      # # print "mu_exp:\n", mu_exp
      # grad_r = mu_D - mu_exp
      # grad_r = np.reshape(grad_r, (out_shape[0], out_shape[1], 1), order='F')
      # grad_rs.append(grad_r)
    grad_rs = np.array(grad_rs)
    grad_theta, l2_loss, grad_norm = nn_r.apply_grads(feat_maps, grad_rs, is_train=True)
    if itr == 0 or (itr + 1) % 20 == 0:
      print "grad_mean_8: ", np.mean(grad_rs, axis=0).reshape(np.dot(*out_shape))[::8]
      print "grad_var: ", np.var(grad_rs)
      print "grad_diff: ", np.mean(np.abs(grad_rs))
    if itr==0 or (itr+1)%200==0 or (itr+1)==n_iters:
      saver.save(nn_r.sess, ckpt_path+"/model_{}.ckpt".format(itr))
    print "itr time: ", time.time() - t
    
def test_model(inputs, nn_r, P_a, gamma, trajs, lr, n_iters, gpu_fraction, ckpt_path):
  N_TRAJ = len(inputs)
  out_shape = nn_r.out_shape
  N_STATES, _, N_ACTIONS = np.shape(P_a)
  # init nn model
  nn_r = FCNIRL(inputs.shape[1:], out_shape, lr, 3, 3, gpu_fraction)
  saver = tf.train.Saver(tf.global_variables())
  saver.restore(nn_r.sess, ckpt_path)
  n_rewards = []
  for i in range(N_TRAJ):
    print "VI traj {}".format(i)
    print "traj {}".format(i)
    feat_map, traj = inputs[i], trajs[i]
    feat_map = np.array([feat_map])
    # mu_D = demo_sparse_svf(traj, N_STATES)
    mu_D = demo_svf(traj, N_STATES)
    # print "mu_D:\n", mu_D
    reward = nn_r.get_rewards(feat_map, is_train=False)
    # print "rewards\n", rewards
    n_rewards.append(np.reshape(reward, out_shape[0] * out_shape[1], order='F'))
    reward = np.reshape(reward, N_STATES, order='F')
    # rewards = np.reshape(rewards, feat_map.shape[1]*feat_map.shape[2], order='F')
    _, policy = value_iteration.value_iteration(P_a, reward, gamma, error=0.1, deterministic=True, max_itrs=np.inf)
    mu_exp = compute_state_visition_freq(P_a, gamma, [traj], policy, deterministic=True)
    # print "mu_exp:\n", mu_exp
    grad_r = mu_D - mu_exp
    print "grad_r: \n", grad_r

  return n_rewards


def simple_fcn_maxent_irl(feat_maps, nn_r, P_a, gamma, trajs, lr, n_iters, gpu_fraction, ckpt_path, batch_size=16,
                   max_itr=np.inf, ids=[]):
  N_STATES, _, N_ACTIONS = np.shape(P_a)
  out_shape = nn_r.out_shape
  
  # init nn model
  saver = tf.train.Saver(tf.global_variables())
  itr_interval = n_iters / 20
  for itr in range(n_iters):
    t = time.time()
    print "--------itr {}--------".format(itr)
    grad_rs = []
    for i in range(batch_size):
      feat_map, traj = feat_maps[i], trajs[i]
      feat_map = np.array([feat_map])
      # mu_D = demo_sparse_svf(traj, N_STATES)
      print ids[i]
      mu_D = demo_svf(traj, N_STATES)
      # print "mu_D:\n", mu_D
      reward = nn_r.get_rewards(feat_map, is_train=True)
      # print "rewards\n", rewards
      reward = np.reshape(reward, N_STATES, order='F')
      # rewards = np.reshape(rewards, feat_map.shape[1]*feat_map.shape[2], order='F')
      _, policy = value_iteration.value_iteration(P_a, reward, gamma, error=0.1, deterministic=True, max_itrs=max_itr)
      mu_exp = compute_state_visition_freq(P_a, gamma, [traj], policy, deterministic=True)
      # print "mu_exp:\n", mu_exp
      grad_r = mu_D - mu_exp
      if itr == 0 or (itr + 1) % 20 == 0:
        print "grad_r: ", grad_r
      grad_r = np.reshape(grad_r, (out_shape[0], out_shape[1], 1), order='F')
      grad_rs.append(grad_r)
    grad_rs = np.array(grad_rs)
    grad_theta, l2_loss, grad_norm = nn_r.apply_grads(feat_maps, grad_rs, is_train=True)
    if itr == 0 or (itr + 1) % itr_interval == 0:
      saver.save(nn_r.sess, ckpt_path + "/model_{}.ckpt".format(itr))
    print "itr time: ", time.time() - t
  # rewards = np.reshape(rewards, feat_map.shape[0]*feat_map.shape[1], order='F')
  # norm_rewards = []
  # for reward, feat_map in zip(rewards, feat_maps):
  #   reward = np.reshape(reward, feat_map.shape[0] * feat_map.shape[1], order='F')
  #   norm_reward = normalize(reward)
  #   norm_rewards.append(norm_reward)
  # return norm_rewards
