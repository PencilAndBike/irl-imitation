import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import namedtuple

import img_utils
from mdp import gridworld
from mdp import value_iteration
from fcn_maxent_irl import *
from maxent_irl import *
from utils import *
import os
import time
import logging

Step = namedtuple('Step','cur_state action next_state reward done')

PARSER = argparse.ArgumentParser(description=None)
PARSER.add_argument('-hei', '--height', default=5, type=int, help='height of the gridworld')
PARSER.add_argument('-wid', '--width', default=5, type=int, help='width of the gridworld')
PARSER.add_argument('-g', '--gamma', default=0.9, type=float, help='discount factor')
PARSER.add_argument('-a', '--act_random', default=0.0, type=float, help='probability of acting randomly')
PARSER.add_argument('-t', '--n_trajs', default=5, type=int, help='number of expert trajectories')
PARSER.add_argument('-l', '--l_traj', default=20, type=int, help='length of expert trajectory')
PARSER.add_argument('--rand_start', dest='rand_start', action='store_true', help='when sampling trajectories, randomly pick start positions')
PARSER.add_argument('--no-rand_start', dest='rand_start',action='store_false', help='when sampling trajectories, fix start positions')
PARSER.set_defaults(rand_start=True)
PARSER.add_argument('-lr', '--learning_rate', default=0.02, type=float, help='learning rate')
PARSER.add_argument('-ni', '--n_iters', default=30, type=int, help='number of iterations')
PARSER.add_argument('-sd', '--save_dir', default="./exps", type=str, help='save dir')
PARSER.add_argument('-name', '--exp_name', default="gw10_fcn_sparse_feed", type=str, help='experiment name')
PARSER.add_argument('-n_exp', '--n_exp', default=20, type=int, help='repeat experiment n times')
PARSER.add_argument('-gpu_frac', '--gpu_fraction', default=0.4, type=float, help='gpu fraction')
PARSER.add_argument('-term', '--terminal', default=True, type=bool, help='terminal or not when agent reach the goal')
ARGS = PARSER.parse_args()
print ARGS


GAMMA = ARGS.gamma
ACT_RAND = ARGS.act_random
R_MAX = 1 # the constant r_max does not affect much the recoverred reward distribution
H = ARGS.height
W = ARGS.width
N_TRAJS = ARGS.n_trajs
L_TRAJ = ARGS.l_traj
RAND_START = ARGS.rand_start
LEARNING_RATE = ARGS.learning_rate
N_ITERS = ARGS.n_iters
SAVE_DIR = ARGS.save_dir
EXP_NAME = ARGS.exp_name
N_EXP = ARGS.n_exp
GPU_FRACTION = ARGS.gpu_fraction
TERMINAL = ARGS.terminal

class GWExperiment(object):
  def __init__(self, gamma=0.9, act_rand=0.3, r_max=1, h=10, w=10, n_trajs=100, l_traj=20, rand_start=True,
               learning_rate=0.02, n_iters=20, save_dir="./exps", exp_name="gw_"+str(int(time.time())),
               n_exp=20, feat_map=None, gpu_fraction=0.2, terminal=True):
    self._gamma, self._act_rand, self._r_max, self._h, self._w, self._n_trajs, self._l_traj, self._rand_start, \
    self._learning_rate, self._n_iters, self._save_dir, self._exp_name, self._n_exp = \
      gamma, act_rand, r_max, h, w, n_trajs, l_traj, rand_start, learning_rate, n_iters, save_dir, exp_name, n_exp
    save_dir_exp = save_dir+"/"+exp_name
    if not os.path.exists(save_dir_exp):
      os.makedirs(save_dir_exp)
    # else:
    #   logging.warning(save_dir_exp)
    #   exit()
    n = len(filter(lambda x: '.' not in x, os.listdir(save_dir_exp)))
    self._exp_result_path = save_dir_exp+"/"+str(n+1)
    os.makedirs(self._exp_result_path)
    self._gpu_fraction = gpu_fraction
    # rmap_gt = np.zeros([h, w])
    # rmap_gt[h-1, w-1] = rmap_gt[0, w-1] = rmap_gt[h-1, 0] = r_max
    # if terminal:
    #   self._gw = gridworld.GridWorld(rmap_gt, {(h-1, w-1), (0, w-1), (h-1, 0)}, 1 - ACT_RAND)
    # else:
    #   self._gw = gridworld.GridWorld(rmap_gt, {}, 1 - ACT_RAND)
    # self._rewards_gt = np.reshape(rmap_gt, **W, order='F')
    # self._P_a = self._gw.get_transition_mat()
    # ts = time.time()
    # self._values_gt, self._policy_gt = value_iteration.value_iteration(self._P_a, self._rewards_gt, GAMMA, error=0.01,
    #                                                                    deterministic=True)
    # te = time.time()
    # print "value iteration time of ground truth: ", te-ts
    # ts = time.time()
    # self.save_plt("gt", (3*w, h), self._rewards_gt, self._values_gt, self._policy_gt)
    # te = time.time()
    # print "saving plt time: ", te-ts
    # self._demo_trajs = self.generate_demonstrations()
    # self._feat_map = rmap_gt.reshape((h,w,1)) if feat_map is None else feat_map
    
  def save_plt(self, name, figsize, rewards, values, policy):
    plt.figure(figsize=figsize)
    plt.subplot(1,3,1)
    img_utils.heatmap2d(np.reshape(rewards, (self._h, self._w), order='F'), 'Rewards Map', block=False)
    plt.subplot(1,3,2)
    img_utils.heatmap2d(np.reshape(values, (self._h, self._w), order='F'), 'Value Map', block=False)
    plt.subplot(1,3,3)
    img_utils.heatmap2d(np.reshape(policy, (self._h, self._w), order='F'), 'Policy Map', block=False)
    plt.savefig(self._exp_result_path+"/"+name+".png")
    plt.close()

  def rnd_gw(self):
    rmap_gt = np.zeros((self._h, self._w))
    mark_pos = (np.random.randint(self._h), np.random.randint(self._w))
    while mark_pos == (self._h/2, self._w/2):
      mark_pos = (np.random.randint(self._h), np.random.randint(self._w))
    rmap_gt[mark_pos] = 1.0
    feat_map = rmap_gt.reshape((self._h,self._w,1))
    gw = gridworld.GridWorld(rmap_gt, {mark_pos}, 1 - self._act_rand)
    return gw, feat_map, rmap_gt

  def generate_demonstrations(self, exp_id):
    trajs = []
    feat_maps = []
    P_as = []
    start_pos = (self._h/2, self._w/2)
    for i in range(self._n_trajs):
      gw, feat_map, reward_gt = self.rnd_gw()
      P_a = gw.get_transition_mat()
      feat_maps.append(feat_map)
      P_as.append(P_a)
      value_gt, policy_gt = value_iteration.value_iteration(P_a, np.reshape(reward_gt, self._h*self._w, order='F'),
                                                            GAMMA, error=0.01, deterministic=True)
      self.save_plt(exp_id+'/'+str(i)+'_gt', (3 * self._w, self._h), reward_gt, value_gt, policy_gt)
      gw.reset(start_pos)
      cur_state = start_pos
      episode = []
      for _ in range(self._l_traj):
        cur_state, action, next_state, reward, is_done = gw.step(int(policy_gt[gw.pos2idx(cur_state)]))
        episode.append(
          Step(cur_state=gw.pos2idx(cur_state), action=action, next_state=gw.pos2idx(next_state), reward=reward,
               done=is_done))
        if is_done:
          break
        cur_state = next_state
      trajs.append(episode)
    return np.array(feat_maps), P_as, trajs

  def test_once(self, exp_id):
    tf.reset_default_graph()
    feat_maps, P_as, demo_trajs = self.generate_demonstrations(exp_id)
    rewards = fcn_maxent_irl(feat_maps, P_as, GAMMA, demo_trajs, self._learning_rate, self._n_iters, self._gpu_fraction)
    for i in range(self._n_trajs):
      P_a = P_as[i]
      reward = rewards[i]
      value, policy = value_iteration.value_iteration(P_a, reward, self._gamma, error=0.01, deterministic=True)
      self.save_plt(exp_id+'/'+str(i), (3 * self._w, self._h), reward, value, policy)

  def test_n_times(self, n):
    for i in range(n):
      print "=============test {}=============".format(i)
      os.mkdir(self._exp_result_path+'/'+str(i))
      self.test_once(str(i))


if __name__ == "__main__":
  gwExperiment = GWExperiment(GAMMA, ACT_RAND, R_MAX, H, W, N_TRAJS, L_TRAJ, RAND_START, LEARNING_RATE,
                              N_ITERS, SAVE_DIR, EXP_NAME, N_EXP)
  gwExperiment.test_n_times(20)