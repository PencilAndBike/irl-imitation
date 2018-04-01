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
PARSER.add_argument('-ni', '--n_iters', default=100000, type=int, help='number of iterations')
PARSER.add_argument('-rd', '--record_dir', default="/home/pirate03/Downloads/prediction_data/crop256", type=str, \
                    help='recording data dir')
PARSER.add_argument('-ld', '--log_dir', default="/home/pirate03/Downloads/prediction_data/crop256/exp/67", type=str, \
                    help='training log dir')
PARSER.add_argument('-name', '--exp_name', default="gw10_fcn_sparse_feed", type=str, help='experiment name')
PARSER.add_argument('-n_exp', '--n_exp', default=20, type=int, help='repeat experiment n times')
PARSER.add_argument('-gpu_frac', '--gpu_fraction', default=0.2, type=float, help='gpu fraction')
PARSER.add_argument('-l2', '--l2', default=0., type=float, help='l2 norm')
PARSER.add_argument('-term', '--terminal', default=True, type=bool, help='terminal or not when agent reach the goal')
# PARSER.add_argument('-is_train', '--is_train', default=True, type=bool, help='train or test')
PARSER.add_argument('--train', dest='is_train', action='store_true', help='train or test')
PARSER.add_argument('--test', dest='is_train',action='store_false', help='train or test')
PARSER.add_argument('-max_vi', '--max_vi', default=10000, type=int, help='value iteration max num')
PARSER.set_defaults(is_train=True)
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
LOG_DIR = ARGS.log_dir
EXP_NAME = ARGS.exp_name
N_EXP = ARGS.n_exp
GPU_FRACTION = ARGS.gpu_fraction
L2 = ARGS.l2
TERMINAL = ARGS.terminal
IS_TRAIN = ARGS.is_train
MAX_VI = ARGS.max_vi

class CarIRLExp(object):
  def __init__(self, l_pos, l_traj, n_demos, rec_dir, log_dir, gamma=0.9,
               learning_rate=0.02, n_iters=20,
               gpu_fraction=0.2, l2=0.2, h=20, w=20, img_h=500, img_w=500, max_vi=10000):
    
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
    self._l2 = l2
    self._h = h
    self._w = w
    self._img_h = img_h
    self._img_w = img_w
    # save_dir_exp = self._rec_dir+"/exp"
    # n = len(filter(lambda x: '.' not in x, os.listdir(save_dir_exp)))
    self._log_dir = log_dir
    self._exp_result_path = log_dir+"/result"
    # if not os.path.exists(self._exp_result_path):
    #   os.makedirs(self._exp_result_path)
    self._ckpt_path = log_dir+"/ckpt"
    print "ckpt_path: ", self._ckpt_path
    self._max_vi = max_vi
    # if not os.path.exists(self._ckpt_path):
    #   os.makedirs(self._ckpt_path)

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
  
  
  def dot_pos(self, pre_pos, pos):
    origin = pre_pos + np.array([self._img_h/2, -self._img_w/2]) * 0.2
    pos = (pos-origin)/0.2
    pos = [-pos[0], pos[1]]
    return np.array([int(round(pos[0])), int(round(pos[1]))])

  def render_car(self, x, y, theta, l=24.0, w=12.0):
    pass

  def get_stack_traj(self, i):
    traj = []
    # img1, img2, img3 = self._imgs[i], self._imgs[i+1], self._imgs[i+2]
    img1 = self._imgs[i]
    for j in range(self._l_pos):
      traj.append(self._poses[i+j])
    disc_traj, goal = LinkAndDiscTraj(traj=traj, img_h=self._img_h, img_w=self._img_w,
                                grid_h=self._grid_h, grid_w=self._grid_w, idx=i).discrete()
    idx_traj = []
    for disc_pos in disc_traj:
      idx_traj.append(self._car.pos2idx(disc_pos))
    
    # Currently just use simple square as the goal. In fact, use car shape could be better.
    goal_img = np.zeros([self._img_h, self._img_w, 1])
    goal_img[goal[0]-6:goal[0]+6, goal[1]-6:goal[1]+6, :] = 100
    # goal_img = goal_img[np.newaxis, :, :, np.newaxis]
    img = np.concatenate((img1, goal_img), axis=-1)
    return MTraj(i, img, idx_traj)

  def test_stack_goal_img(self, i):
    img = self._imgs[i]
    traj = []
    for j in range(self._l_pos):
      traj.append(self._poses[i + j])
    disc_traj, goal = LinkAndDiscTraj(traj=traj, img_h=self._img_h, img_w=self._img_w,
                                grid_h=self._grid_h, grid_w=self._grid_w, idx=i).discrete()
    print "goal: ", goal
    idx_traj = []
    for disc_pos in disc_traj:
      idx_traj.append(self._car.pos2idx(disc_pos))

    # Currently just use simple square as the goal. In fact, use car shape could be better.
    # goal = self.dot_pos(traj[0], traj[-1])
    # goal_img = np.zeros(img.shape)
    # goal_img[goal[0] - 10:goal[0] + 10, goal[1] - 10:goal[1] + 10, :] = 255
    # goal_img = goal_img[np.newaxis, :, :, np.newaxis]
    img[max(goal[0]-10,0):goal[0]+10, max(goal[1]-10,0):goal[1]+10, -1] = 255
    cv2.imwrite(self._exp_result_path+"/"+str(i)+".jpg", img)

  def get_demo_trajs(self, ids):
    feat_maps = []
    trajs = []
    for id in ids:
      mtraj = self.get_stack_traj(id)
      feat_maps.append(mtraj.img)
      trajs.append(mtraj.traj)
    return np.array(feat_maps), np.array(trajs)
    
    
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
    
  def train(self, ids=[]):
    print "getting demo trajs..."
    t = time.time()
    inputs, demo_trajs = self.get_demo_trajs(ids)
    print "got demo trajs: {}s".format(time.time()-t)
    wraped_trajs = self.wrap_trajs(demo_trajs)
    print "Run fcn_maxent_irl..."
    nn_r = FCNIRL(inputs.shape[1:], np.array([self._h, self._w]), self._learning_rate, l2=self._l2,
                  gpu_fraction=self._gpu_fraction)
    fcn_maxent_irl(inputs, nn_r, self._P_a, self._gamma, wraped_trajs,
                   self._learning_rate, self._n_iters, self._gpu_fraction, self._ckpt_path, self._n_demos,
                   max_itr=self._max_vi)
    print "fcn_maxent_irl is done"


  def simple_train(self):
    ids = np.random.randint(450, size=self._n_demos)
    # ids = range(448, 455)
    inputs, demo_trajs = self.get_demo_trajs(ids)
    wraped_trajs = self.wrap_trajs(demo_trajs)
    nn_r = FCNIRL(inputs.shape[1:], np.array([self._h, self._w]), self._learning_rate, l2=self._l2,
                  gpu_fraction=self._gpu_fraction)
    simple_fcn_maxent_irl(inputs, nn_r, self._P_a, self._gamma, wraped_trajs,
                   self._learning_rate, self._n_iters, self._gpu_fraction, self._ckpt_path, self._n_demos,
                   max_itr=self._max_vi, ids=ids)
    print "fcn_maxent_irl is done"


  def test(self, ids):
    names = os.listdir(self._log_dir)
    n_results = len(filter(lambda x: 'result' in x, names))
    self._exp_result_path = self._log_dir+"/result"+str(n_results)
    if not os.path.exists(self._exp_result_path):
      os.makedirs(self._exp_result_path)
    inputs, demo_trajs = self.get_demo_trajs(ids)
    trajs = self.wrap_trajs(demo_trajs)
    nn_r = FCNIRL(inputs.shape[1:], np.array([self._h, self._w]), self._learning_rate, l2=self._l2,
                  gpu_fraction=self._gpu_fraction)
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(nn_r.sess, self._ckpt_path+"/model_"+str(self._n_iters-1)+".ckpt")
    N_TRAJ = len(inputs)
    N_STATES, _, N_ACTIONS = np.shape(self._P_a)
    for i in range(N_TRAJ):
      print "run demo {}".format(ids[i])
      feat_map, traj = inputs[i], trajs[i]
      feat_map = np.array([feat_map])
      # mu_D = demo_sparse_svf(traj, N_STATES)
      mu_D = demo_svf(traj, N_STATES)
      # print "mu_D:\n", mu_D
      reward = nn_r.get_rewards(feat_map, is_train=True)
      # print "rewards\n", rewards
      reward = np.reshape(reward, N_STATES, order='F')
      value, policy = value_iteration.value_iteration(self._P_a, reward, self._gamma, error=0.1, deterministic=True,
                                                      max_itrs=self._max_vi)
      mu_exp = compute_state_visition_freq(self._P_a, self._gamma, [traj], policy, deterministic=True)
      # print "mu_exp:\n", mu_exp
      grad_r = mu_D - mu_exp
      print "grad_r: \n", grad_r
      cv2.imwrite(self._exp_result_path + '/' + str(i) + '_' + str(ids[i]) + ".png",
                  inputs[i][:, :, :3])
      self.save_plt(str(i), (4 * self._w, self._h), demo_trajs[i], reward, value, policy)


def test_heatmap():
  x = np.arange(9).reshape((3,3))
  plt.subplot(1,1,1)
  img_utils.heatmap2d(x, 'x', block=False)
  plt.savefig('test.png')
  plt.close()
    
if __name__ == "__main__":
  car_irl_exp = CarIRLExp(L_POS, L_TRAJ, N_DEMOS, RECORD_DIR, LOG_DIR, n_iters=N_ITERS, gpu_fraction=GPU_FRACTION,
                          l2=L2, h=H, w=W, img_h=IH, img_w=IW, max_vi=MAX_VI)
  # car_irl_exp.test(range(450, 500))
  print "IS TRAIN OR NOT: ", IS_TRAIN
  if IS_TRAIN:
    print "we are training"
    # ids = np.random.randint(450, size=32)
    # ids = range(20)
    car_irl_exp.train(range(400))
    # car_irl_exp.simple_train()
  else:
    print "we are testing"
    car_irl_exp.test(range(450))
    # for i in range(30, 50):
    #   car_irl_exp.test_stack_goal_img(i)
  # car_irl_exp.test(range(68, 75)+range(116,126)+range(178,185)+range(450, 460)+range(499,502))
  # ckpt_path = "/home/pirate03/PycharmProjects/irl-imitation/ckpt4/model_1499.ckpt"
  # car_irl_exp.test('test', ckpt_path, ids=[3, 4, 5, 6])
  # test_heatmap()
