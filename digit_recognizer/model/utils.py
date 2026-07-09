import numpy as np


LEARN_RATE = .1


def mse_loss(y_true, y_pred):
  return ((y_true - y_pred) ** 2).mean()

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(fx):
  return fx * (1 - fx)
