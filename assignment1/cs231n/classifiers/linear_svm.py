import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """                                                                                           Structured SVM loss function, naive implementation (with loops).                             Inputs have dimension D, there are C classes, and we operate on minibatches                  of N examples.                                                                              
   Inputs:                                                                                      - W: A numpy array of shape (D, C) containing weights.                                       - X: A numpy array of shape (N, D) containing a minibatch of data.                           - y: A numpy array of shape (N,) containing training labels; y[i] = c means                    that X[i] has label c, where 0 <= c < C.                                                   - reg: (float) regularization strength                                                                                                                                                    Returns a tuple of:                                                                           - loss as single float                                                                       - gradient with respect to weights W; an array of same shape as W                           
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero                                    
  # compute the loss and the gradient                                                        
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    # print "scores", scores
    correct_class_score = scores[y[i]]
    margins = np.zeros(W.shape[1])
    count_margins_gt_zero = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margins[j] = scores[j] - correct_class_score + 1 # note delta = 1 
      # print "margin[%d] = %s" % (j , margins[j])
      if margins[j] < 0:
            margins[j] = 0
      if margins[j] > 0:
        dW[:, j] += X[i].T
        count_margins_gt_zero += 1
    loss += np.sum(margins)
    dW[:, y[i]] -= count_margins_gt_zero * X[i].T
    
  # Right now the loss is a sum over all training examples, but we want it                     # to be an average instead so we divide by num_train.                                       
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.                                                            loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  num_dim = W.shape[0]
  num_class = W.shape[1]
  num_train = X.shape[0]

  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  S = X.dot(W)
  y_i = np.choose(y, S.T)
  y_i = np.reshape(y_i, (S.shape[0], 1))
  M = S - y_i + 1
  for row in xrange(M.shape[0]):
    M[row, y[row]] = 0
  M = np.maximum(np.zeros(M.shape), M)
  loss = np.sum(M)
  loss /= num_train

  # regularization
  loss += 0.5 * reg * np.sum(W * W)

  C = np.sign(M)
  for i in xrange(num_train):
    # not Wyi
    x_i = np.reshape(X[i], (num_dim, 1))
    c_i = np.reshape(C[i], (1, num_class))
    dW_i = x_i.dot(c_i) 

    #Wyi
    dW_i[:, y[i]] = -np.sum(c_i) * x_i[:, 0] 

    # accumulate over num_train
    dW += dW_i

  dW /= num_train 
  dW += reg * W
 
  return loss, dW
