from collections import namedtuple
import numpy as np
import tensorflow as tf
import random

Step = namedtuple('Step', 'pos next_pos reward')
MTraj = namedtuple('MTraj', 'state_map traj')
# MTraj = namedtuple('MTraj', 'stacking_state_map traj')


class Car(object):
  def __init__(self, f_dir, l):
    self.imgs = []
    self.poses = []
    self._L = len(self.imgs)
    self._l = l
    pass
  
  # def getNearPos(self, pos):
  #   pass

  @staticmethod
  def step(smap, rmap, policy, pos):
    """
    :param smap:
    :param rmap:
    :param policy:
    :param pos:
    :return:
    """
    # near_positions= Car.getNearPos(pos)
    # near_values = []
    # for pos in near_positions:
    #   near_values.append(vmap[pos])
    # max_i = np.argmax(near_values)
    next_pos = policy(smap, pos)
    reward = rmap[pos]
    return Step(pos, next_pos, reward)
    
  @staticmethod
  def step_n(smap, rmap, policy, pos, l):
    """
    make agent take l steps and return the traj
    :param self:
    :param vmap:
    :param pos:
    :param l:
    :return:
    """
    traj = []
    for _ in range(l):
      next_pos, reward = Car.step(smap, rmap, policy, pos)
      traj.append(Step(pos, next_pos, reward))
      pos = next_pos
    return traj
  
  
  def rand_traj(self):
    i = random.randint(self._L - self._l)
    traj = []
    img = self.imgs[i]
    for j in range(self._l):
      traj.append(Step(self.poses[i+j], self.poses[i+j+1], None))
    return MTraj(img, traj)
  
  def get_demo_trajs(self, n):
    trajs = []
    for _ in range(n):
      trajs.append(self.rand_traj())
    return trajs
  

class Policy(object):
  def __init__(self):
    pass
  
  def act(self, smap, pos):
    pass

    
    
    
    

