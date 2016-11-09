import numpy as np
from random import shuffle

def loss_i(W, x, y):
    """
    Calculate the loss for a single point
    W CxD
    x Dx1
    y 1
    """
    loss = 0
    f = W.dot(x)
    f -= np.max(f)
    loss = -f[y] + np.log(np.sum(np.exp(f)))
    return loss

def loss_n(W, X, y, reg):
    """
    W CxD
    x DxN
    y N
    """
    n_points = X.shape[1]
    loss = 0
    for point in range(n_points):
        loss += loss_i(W, X[:, point], y[point])
    loss /= n_points
    loss += 0.5 * reg * np.sum(W * W)
    return loss

def gradient_i(W, x, y):
    """
    W CxD
    x Dx1
    y N
    """
    f = W.dot(x)
    f -= np.max(f)
    dW = np.zeros_like(W)
    dW[y] -= x
    sum_f = np.sum(np.exp(f))
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            dW[i, j] += np.exp(f[i]) * x[j] / sum_f
    return dW
               
def gradient_n(W, X, y, reg):
    """
    W CxD
    x DxN
    y N
    """
    dW = np.zeros_like(W)
    n_points = X.shape[1]
    for point in range(n_points):
        dW += gradient_i(W, X[:, point], y[point])
    dW /= n_points
    dW += reg * W
    return dW


def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """




  loss = loss_n(W.T, X.T, y, reg)
  dW = gradient_n(W.T, X.T, y, reg).T



  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  W = W.T
  X = X.T
  n_points = X.shape[1]

  loss = 0
  F = W.dot(X)
  F -= np.max(F, axis=0)
  F_exp = np.exp(F)

  loss = -np.sum(F[y, np.arange(n_points)]) 
  loss += np.sum(np.log(np.sum(F_exp, axis=0)))
  loss /= n_points
  loss += 0.5 * reg * np.sum(W * W)  

  dW = np.zeros_like(W)
  for i in range(dW.shape[0]):
      dW[i] = -np.sum(X.T[y == i], axis=0)

  F_exp_over_sum = F_exp / np.sum(F_exp, axis=0)
  dW += F_exp_over_sum.dot(X.T)
  dW /= n_points
  dW += reg * W

  return loss, dW.T


