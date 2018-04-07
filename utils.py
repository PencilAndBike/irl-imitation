import numpy as np
import math
from collections import namedtuple

Step = namedtuple('Step','cur_state action next_state reward done')


def normalize(vals):
  """
  normalize to (0, max_val)
  input:
    vals: 1d array
  """
  min_val = np.min(vals)
  max_val = np.max(vals)
  return (vals - min_val + 1e-5) / (max_val - min_val + 1e-5)


def sigmoid(xs):
  """
  sigmoid function
  inputs:
    xs      1d array
  """
  return [1 / (1 + math.exp(-x)) for x in xs]