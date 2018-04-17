from collections import namedtuple
from collections import namedtuple
import numpy as np
import tensorflow as tf
import random

Step = namedtuple('Step', 'pos next_pos reward')
MTraj = namedtuple('MTraj', 'img traj')
# MTraj = namedtuple('MTraj', 'stacking_state_map traj')

# def discrete(img, pos, next_pos):
#   move = (next_pos-pos)/0.2
#   return move


class LinkAndDiscTraj(object):
  def __init__(self, traj, img_h=500, img_w=500, grid_h=25, grid_w=25, particle=0.2, idx=None):
    """
    :param traj: np.array [pos1, pos2, ..., pos_n]
    :param h:
    :param w:
    :param particle:
    """
    self._traj = traj
    self._start_pos = np.copy(traj[0])
    self._img_h = img_h
    self._img_w = img_w
    self._grid_h = grid_h
    self._grid_w = grid_w
    self._particle = particle
    self._origin = self._start_pos + np.array([-img_h/2, -img_w/2])*particle
    self._idx = idx

    
  def link_pos(self, pos, next_pos):
    """
    Sometimes some position is repeated.
    :param pos:
    :param next_pos:
    :return:
    """
    n = int(np.ceil(np.sqrt(np.sum((pos-next_pos)**2))/self._particle))
    if n == 0:
      return [np.copy(pos)]
    delta_x = (next_pos[0]-pos[0])/n
    delta_y = (next_pos[1]-pos[1])/n
    inter_pos = np.copy(pos)
    linked_traj = [np.copy(inter_pos)]
    for i in range(n):
      inter_pos[0] += delta_x
      inter_pos[1] += delta_y
      linked_traj.append(np.copy(inter_pos))
    return linked_traj
  
  
  def dot_pos(self, pos):
    
    pos = (pos-self._origin)/self._particle
    pos = [pos[0], pos[1]]
    return np.array([int(round(pos[0])), int(round(pos[1]))])
    # return disc_pos / np.array([self._h, self._w])
  

  def dot_traj(self, traj):
    doted_traj = []
    for pos in traj:
      # To discriminate if doted_pos is in doted_traj, it has to be tupled.
      # Otherwise directly using numpy.array, it would raise an error:
      # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
      doted_pos = tuple(self.dot_pos(pos))
      # To avoid position out of index
      if doted_pos[0]<0 or doted_pos[1]<0 or doted_pos[0]>self._img_h-1 or doted_pos[1]>self._img_w-1:
        break
      if doted_pos in doted_traj:
        pass
      else:
        doted_traj.append(doted_pos)
    return np.array(doted_traj)
  
  
  def dot(self):
    linked_traj = []
    for i in range(len(self._traj)-1):
      pos = self._traj[i]
      next_pos = self._traj[i+1]
      try:
        inter_linked_traj = self.link_pos(pos, next_pos)
      except Exception as e:
        print e
        print self._idx
        inter_linked_traj = []
      finally:
        linked_traj.extend(inter_linked_traj)
    
    # calculate the goal line
    pos = self._traj[-3]
    next_pos = self._traj[-1]
    road_width = 12.5
    if pos[1] == next_pos[1]:
      f_pos1 = np.array(next_pos[0], next_pos[1]-road_width/2)
      f_pos2 = np.array(next_pos[0], next_pos[1]+road_width/2)
    else:
      # xielv = (next_pos[1] - pos[1]) / (next_pos[0] - pos[0])
      # f_xielv = -1.0 / xielv
      f_xielv = - (next_pos[0]-pos[0])/(next_pos[1]-pos[1])
      l_grid = np.sqrt(1+f_xielv**2)
      delta = np.array([road_width/2/l_grid, road_width/2/l_grid*f_xielv])
      f_pos1 = next_pos + delta
      f_pos2 = next_pos - delta
    goal_line = self.link_pos(f_pos1, f_pos2)
    
    return self.dot_traj(linked_traj), self.dot_traj(goal_line)
    
  
  def is_tile(self, pos1, pos2):
    tile_poses = [(pos1[0]-1, pos1[1]-1),
                  (pos1[0]-1, pos1[1]+1),
                  (pos1[0]+1, pos1[1]-1),
                  (pos1[0]+1, pos1[1]+1)]
    return True if pos2 in tile_poses else False
  
  
  def tile_traj(self, traj):
    tiled_traj = [tuple(traj[0])]
    i = 0
    while i < len(traj)-2:
      if self.is_tile(traj[i], traj[i+2]):
        tiled_traj.append(traj[i+2])
        i += 2
      else:
        tiled_traj.append(traj[i+1])
        i += 1
    if i == len(traj)-2:
      tiled_traj.append(traj[-1])
    return tiled_traj
    
    
  def distile_traj(self, traj):
    distiled_traj = [tuple(traj[0])]
    for i in range(1, len(traj)):
      x0, y0 = distiled_traj[-1]
      x, y = traj[i]
      if (x0+1, y0-1) == (x, y) or (x0+1, y0+1) == (x, y):
        distiled_traj.append((x0+1, y0))
      if (x0-1, y0-1) == (x, y) or (x0-1, y0+1) == (x, y):
        distiled_traj.append((x0-1, y0))
      distiled_traj.append((x, y))
    return distiled_traj
    
  def discrete_pos(self, pos):
    return pos / np.array([self._grid_h, self._grid_w])
  
  def discrete(self):
    doted_traj, doted_goal_lin = self.dot()
    goal = doted_traj[-1]
    goal = np.array([goal[0], goal[1]])
    # doted_traj = self.tile_traj(doted_traj)
    discreted_traj = []
    for dot_pos in doted_traj:
      # To discriminate if discreted_pos is in discreted_traj, it has to be tupled.
      # Otherwise directly using numpy.array, it would raise an error:
      # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
      discreted_pos = tuple(self.discrete_pos(dot_pos))
      if discreted_pos not in discreted_traj:
        discreted_traj.append(discreted_pos)
    # discreted_traj = self.distile_traj(discreted_traj)
    # discreted_traj = self.tile_traj(discreted_traj)
    return discreted_traj, goal, doted_goal_lin
  


class Car(object):
  def __init__(self, height=20, width=20, terminals={}):
    self.height = height
    self.width = width
    # self.neighbors = [(-1, -1), (-1, 0), (-1, 1),
    #                   (0, -1), (0, 0), (0, 1),
    #                   (1, -1), (1, 0), (1, 1)]
    # self.actions = [0, 1, 2,
    #                 3, 4, 5,
    #                 6, 7, 8]
    self.neighbors = [(-1, 0),
                      (0, -1), (0, 0), (0, 1),
                      (1, 0)]
    self.actions = [0, 1, 2,
                    3, 4]
    self.n_actions = len(self.actions)
    self.terminals = terminals

  
  def get_actions(self, state):
    """
    get all the actions that can be takens on the current state
    returns
      a list of actions
    """
    actions = []
    for i in range(len(self.actions) - 1):
      inc = self.neighbors[i]
      a = self.actions[i]
      nei_s = (state[0] + inc[0], state[1] + inc[1])
      if nei_s[0] >= 0 and nei_s[0] < self.height and nei_s[1] >= 0 and nei_s[1] < self.width:
        actions.append(a)
    return actions


  def is_terminal(self, state):
    if tuple(state) in self.terminals:
      return True
    else:
      return False
    
  def set_terminal(self, termials):
    self.terminals = termials
    
  
  def get_transition_states_and_probs(self, state, action):
    """
    get all the possible transition states and their probabilities with [action] on [state]
    args
      state     (y, x)
      action    int
    returns
      a list of (state, probability) pair
    """
    if self.is_terminal(tuple(state)):
      return [(tuple(state), 1)]

    inc = self.neighbors[action]
    nei_s = (state[0] + inc[0], state[1] + inc[1])
    if nei_s[0] >= 0 and nei_s[0] < self.height and nei_s[
            1] >= 0 and nei_s[1] < self.width:
      return [(nei_s, 1)]
    else:
      # if the state is invalid, stay in the current state
      return [(state, 1)]

      
  def get_transition_mat(self):
    """
    get transition dynamics of the gridworld

    return:
      P_a         NxNxN_ACTIONS transition probabilities matrix -
                    P_a[s0, s1, a] is the transition prob of
                    landing at state s1 when taking action
                    a at state s0
    """
    N_STATES = self.height*self.width
    N_ACTIONS = len(self.actions)
    P_a = np.zeros((N_STATES, N_STATES, N_ACTIONS))
    for si in range(N_STATES):
      posi = self.idx2pos(si)
      for a in range(N_ACTIONS):
        probs = self.get_transition_states_and_probs(posi, a)

        for posj, prob in probs:
          sj = self.pos2idx(posj)
          # Prob of si to sj given action a
          P_a[si, sj, a] = prob
    return P_a


  def pos2idx(self, pos):
    """
    input:
      column-major 2d position
    returns:
      1d index
    """
    return pos[0] + pos[1] * self.height

  def idx2pos(self, idx):
    """
    input:
      1d idx
    returns:
      2d column-major position
    """
    return (idx % self.height, idx / self.height)

    
  # def rand_traj(self):
  #   i = random.randint(self._L - self._l)
  #   traj = []
  #   img = self._imgs[i]
  #   for j in range(self._l):
  #     traj.append(self._poses[i+j])
  #   disc_traj = LinkAndDiscTraj(traj=traj).discrete()
  #   return MTraj(img, disc_traj)
  
  # def get_demo_trajs(self, n):
  #   imgs = []
  #   trajs = []
  #   for _ in range(n):
  #     mtraj = self.rand_traj()
  #     imgs.append(mtraj.img)
  #     trajs.append(mtraj.traj)
  #   return imgs, trajs
  #
