import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import namedtuple
import img_utils
from mdp import value_iteration
from fcn_maxent_irl import *
from maxent_irl import *
from utils import *
from mdp.car import *
import os
import time
import logging
from mdp.img_process import ProcessCarImg
import cv2

Step = namedtuple('Step','cur_state action next_state reward done')

PARSER = argparse.ArgumentParser(description=None)
PARSER.add_argument('-hei', '--height', default=16, type=int, help='height of the grid map')
PARSER.add_argument('-wid', '--width', default=16, type=int, help='width of the grid map')
PARSER.add_argument('-img_hei', '--img_height', default=256, type=int, help='height of the img')
PARSER.add_argument('-img_wid', '--img_width', default=256, type=int, help='width of the img')
PARSER.add_argument('-g', '--gamma', default=0.9, type=float, help='discount factor')
PARSER.add_argument('-nd', '--n_demos', default=16, type=int, help='number of expert trajectories')
PARSER.add_argument('-lp', '--l_pos', default=7, type=int, help='length of concated positions')
PARSER.add_argument('-lt', '--l_traj', default=10, type=int, help='length of discrete trajectory')
PARSER.add_argument('-lr', '--learning_rate', default=0.02, type=float, help='learning rate')
PARSER.add_argument('-ni', '--n_iters', default=1500, type=int, help='number of iterations')
PARSER.add_argument('-rd', '--record_dir', default="/home/pirate03/Downloads/prediction_data/crop256", type=str, \
                    help='recording data dir')
PARSER.add_argument('-name', '--exp_name', default="gw10_fcn_sparse_feed", type=str, help='experiment name')
PARSER.add_argument('-n_exp', '--n_exp', default=20, type=int, help='repeat experiment n times')
PARSER.add_argument('-gpu_frac', '--gpu_fraction', default=0.4, type=float, help='gpu fraction')
PARSER.add_argument('-term', '--terminal', default=True, type=bool, help='terminal or not when agent reach the goal')
ARGS = PARSER.parse_args()
print ARGS


GAMMA = ARGS.gamma
H = ARGS.height
W = ARGS.width
IH = ARGS.img_height
IW = ARGS.img_width
N_DEMOS = ARGS.n_demos
L_POS = ARGS.l_pos
L_TRAJ = ARGS.l_traj
LEARNING_RATE = ARGS.learning_rate
N_ITERS = ARGS.n_iters
RECORD_DIR = ARGS.record_dir
EXP_NAME = ARGS.exp_name
N_EXP = ARGS.n_exp
GPU_FRACTION = ARGS.gpu_fraction
TERMINAL = ARGS.terminal


class CarIRLExp(object):
  def __init__(self, l_pos, l_traj, n_demos, rec_dir, gamma=0.9,
               learning_rate=0.02, n_iters=20,
               gpu_fraction=0.2, h=20, w=20, img_h=500, img_w=500):
    
    self._rec_dir = rec_dir
    self._imgs = self.get_imgs(rec_dir)
    self._poses = self.get_poses(rec_dir)
    self._L = len(self._imgs)
    self._l_pos = l_pos
    self._l_traj = l_traj
    self._n_demos = n_demos
    self._grid_h, self._grid_w = img_h/h, img_w/w
    self._car = Car(rec_dir, l_pos, h, w)
    print "getting car's transition mat..."
    self._P_a = self._car.get_transition_mat()
    print "got car's transition mat"
    self._gamma = gamma
    self._learning_rate = learning_rate
    self._n_iters = n_iters
    self._gpu_fraction = gpu_fraction
    self._h = h
    self._w = w
    self._img_h = img_h
    self._img_w = img_w
    save_dir_exp = self._rec_dir+"/exp"
    n = len(filter(lambda x: '.' not in x, os.listdir(save_dir_exp)))
    self._exp_result_path = save_dir_exp+"/"+str(n)
    os.makedirs(self._exp_result_path)


  @staticmethod
  def get_imgs(rec_dir):
    return ProcessCarImg(rec_dir).concat()
    

  def get_poses(self, rec_dir):
    status_path = rec_dir+'/status.txt'
    poses = []
    with open(status_path) as f:
      lines = f.readlines()
      for i, line in enumerate(lines):
        info = map(float, line.split())
        # poses.append([info[1], info[2], info[4]])
        poses.append([info[2], info[1]])
    return np.array(poses)


  def rand_traj(self):
    i = np.random.randint(self._L - self._l_pos)
    traj = []
    img = self._imgs[i]
    for j in range(self._l_pos):
      traj.append(self._poses[i+j])
    disc_traj = LinkAndDiscTraj(traj=traj, img_h=self._img_h, img_w=self._img_w,
                                grid_h=self._grid_h, grid_w=self._grid_w).discrete()
    idx_traj = []
    for disc_pos in disc_traj:
      idx_traj.append(self._car.pos2idx(disc_pos))
    return MTraj(i, img, idx_traj)
    # return MTraj(img, disc_traj)

  def get_traj(self, i):
    traj = []
    img = self._imgs[i]
    for j in range(self._l_pos):
      traj.append(self._poses[i+j])
    disc_traj = LinkAndDiscTraj(traj=traj, img_h=self._img_h, img_w=self._img_w,
                                grid_h=self._grid_h, grid_w=self._grid_w, idx=i).discrete()
    idx_traj = []
    for disc_pos in disc_traj:
      idx_traj.append(self._car.pos2idx(disc_pos))
    return MTraj(i, img, idx_traj)

  def get_stack_traj(self, i):
    traj = []
    img1, img2, img3 = self._imgs[i], self._imgs[i+1], self._imgs[i+2]
    img = np.concatenate((img1,img2,img3), axis=-1)
    for j in range(self._l_pos):
      traj.append(self._poses[i+j])
    disc_traj = LinkAndDiscTraj(traj=traj, img_h=self._img_h, img_w=self._img_w,
                                grid_h=self._grid_h, grid_w=self._grid_w, idx=i).discrete()
    idx_traj = []
    for disc_pos in disc_traj:
      idx_traj.append(self._car.pos2idx(disc_pos))
    return MTraj(i, img, idx_traj)


  def get_demo_trajs(self, ids=[9, 20, 23, 68, 118]):
    feat_maps = []
    trajs = []
    for id in ids:
      # mtraj = self.rand_traj()
      mtraj = self.get_stack_traj(id)
      # ids.append(mtraj.id)
      feat_maps.append(mtraj.img)
      trajs.append(mtraj.traj)
    return np.array(ids), np.array(feat_maps), np.array(trajs)


  def rand_demo_trajs(self):
    ids = np.random.randint(self._L-self._l_pos-5, size=self._n_demos)
    feat_maps = []
    trajs = []
    for id in ids:
      mtraj = self.get_stack_traj(id)
      feat_maps.append(mtraj.img)
      trajs.append(mtraj.traj)
    return np.array(ids), np.array(feat_maps), np.array(trajs)
    
    
  def save_plt(self, name, figsize, traj, rewards, values, policy):
    plt.figure(figsize=figsize)
    x = np.zeros((self._h, self._w))
    for idx in traj:
      x[self._car.idx2pos(idx)] = 1.0
    # print "traj:\n", traj
    pos_traj = [self._car.idx2pos(idx) for idx in traj]
    # print "pos traj:\n", pos_traj
    plt.subplot(1,4,1)
    img_utils.heatmap2d(x, 'Traj', block=False)
    # print "rewards:\n", np.reshape(rewards, (self._h, self._w), order='F')
    plt.subplot(1,4,2)
    img_utils.heatmap2d(np.reshape(rewards, (self._h, self._w), order='F'), 'Rewards Map', block=False)
    # print "value:\n", np.reshape(values, (self._h, self._w), order='F')
    plt.subplot(1,4,3)
    img_utils.heatmap2d(np.reshape(values, (self._h, self._w), order='F'), 'Value Map', block=False)
    # print "policy:\n", np.reshape(policy, (self._h, self._w), order='F')
    plt.subplot(1,4,4)
    img_utils.heatmap2d(np.reshape(policy, (self._h, self._w), order='F'), 'Policy Map', block=False)
    plt.savefig(self._exp_result_path+"/"+name+".png")
    plt.close()

    
  def shorten_trajs(self, trajs):
    l = min(map(len, trajs))
    shortened_trajs = []
    for traj in trajs:
      shortened_trajs.append(traj[:l])
    # for traj in trajs:
    #   shortened_traj = self.shorten(traj)
    #   if len(shortened_traj) == self._l_traj:
    #     shortened_trajs.append(shortened_traj)
    return shortened_trajs
    
  
  def wrap(self, traj):
    step_wraped_traj = []
    for i in range(len(traj)-1):
      pos, next_pos = traj[i], traj[i+1]
      step_wraped_traj.append(Step(pos, None, next_pos, None, None))
    return step_wraped_traj

  
  def wrap_trajs(self, trajs):
    step_wraped_trajs = []
    for traj in trajs:
      step_wraped_trajs.append(self.wrap(traj))
    return step_wraped_trajs
    
    
  def run(self, P_a, start_idx, policy, l):
    traj = []
    idx = start_idx
    traj.append(idx)
    for i in range(l):
      act = int(policy[idx])
      next_idx = np.argmax(P_a[idx, :, act])
      traj.append(next_idx)
      idx = next_idx
    pos_traj = [self._car.idx2pos(idx) for idx in traj]
    return pos_traj
    
  def test_once(self, exp_id=None):
    os.mkdir(self._exp_result_path+'/'+exp_id)
    print "getting demo trajs..."
    # ids, feat_maps, demo_trajs = self.get_demo_trajs()
    ids, feat_maps, demo_trajs = self.rand_demo_trajs()
    print "got demo trajs"
    # shortened_trajs = self.shorten_trajs(demo_trajs)
    wraped_trajs = self.wrap_trajs(demo_trajs)
    # wraped_trajs =
    #  self.wrap_trajs(demo_trajs)
    # P_as = []
    # for i in range(self._n_demos):
    #   shortened_traj = shortened_trajs[i]
      # terminals = {self._car.idx2pos(shortened_traj[-1])}
      # print "terminals: ", terminals
      # self._car.set_terminal(terminals)
      # P_as.append(self._car.get_transition_mat())
    P_as = [self._P_a] * len(wraped_trajs)
    print "Run fcn_maxent_irl..."
    rewards = fcn_maxent_irl(feat_maps, np.array([self._h, self._w]), P_as, self._gamma, wraped_trajs,
                             self._learning_rate, self._n_iters, self._gpu_fraction)
    print "fcn_maxent_irl is done"
    for i in range(len(wraped_trajs)):
      print "run demo {}".format(i)
      P_a = P_as[i]
      reward = rewards[i]
      value, policy = value_iteration.value_iteration(P_a, reward, self._gamma, error=0.01, deterministic=True)
      # traj = demo_trajs[i]
      # irl_traj = self.run(P_a, traj[0], policy, 2)
      # print "agent run: ", irl_traj
      
      cv2.imwrite(self._exp_result_path+"/"+exp_id+'/'+str(i)+'_'+str(ids[i])+".png", feat_maps[i][:,:,:3])
      self.save_plt(exp_id+'/'+str(i), (4 * self._w, self._h), demo_trajs[i], reward, value, policy)


def test_heatmap():
  x = np.arange(9).reshape((3,3))
  plt.subplot(1,1,1)
  img_utils.heatmap2d(x, 'x', block=False)
  plt.savefig('test.png')
  plt.close()
    
if __name__ == "__main__":
  car_irl_exp = CarIRLExp(L_POS, L_TRAJ, N_DEMOS, RECORD_DIR, n_iters=N_ITERS, h=H, w=W, img_h=IH, img_w=IW)
  car_irl_exp.test_once('1')