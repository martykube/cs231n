import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  (conv - sbn - relu - conv - sbn - relu - 2x2 max pool) - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################

    # Layer parameters
    conv_params = {'stride': 1, 'pad': (filter_size - 1) / 2}
    pool_params = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    bn_params = {}

    # input channels, height, width
    C, H, W = input_dim
    F = num_filters

    #
    # first conv => sbn => relu
    #

    # IN(N, C, H, W) => CONV(N, F, H, W)
    # Use padding at (filter_size -1) / 2 and output dimensions are conserved...
    self.params['W1'] = weight_scale * \
                        np.random.randn(F, C, filter_size, filter_size)
    self.params['b1'] = np.zeros(F)
    self.conv_params_1 = conv_params.copy()

    # Spatial batch normalization
    # CONV(N, F, H, W) =>  SBN(N, F, H, W)
    self.params['gamma1'] = np.ones(F)
    self.params['beta1'] = np.zeros(F)
    self.bn_params_1 = bn_params.copy()

    # SBN(N, F, H, W) => RELU(N, F, H, W)
    
    #
    # second conv => sbn => relu
    #

    # IN(N, F, H, W) => CONV(N, F, H, W)
    # Use padding at (filter_size -1) / 2 and output dimensions are conserved...
    self.params['W2'] = weight_scale * \
                        np.random.randn(F, F, filter_size, filter_size)
    self.params['b2'] = np.zeros(F)
    self.conv_params_2 = conv_params.copy()

    # Spatial batch normalization
    # CONV(N, F, H, W) =>  SBN(N, F, H, W)
    self.params['gamma2'] = np.ones(F)
    self.params['beta2'] = np.zeros(F)
    self.bn_params_2 = bn_params.copy()

    # SBN(N, F, H, W) => RELU(N, F, H, W)
    
    # maxpool
    #
    # RELU(N, F, H, W) => MAXPOOL(N, F, HP, WP)
    # pool_params = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    # (H - F) / S + 1
    HP = (H - 2) / 2 + 1
    WP = (W - 2) / 2 + 1
    self.pool_params = pool_params.copy()

    # MAXPOOL(N, F, HP, WP) => AFFINE(N, NHID)
    # W(F * HP * WP, NHID), seems backwards, I know...
    # affine will do the flattening
    self.params['W3'] = weight_scale * np.random.randn(F * HP * WP, hidden_dim)
    self.params['b3'] = np.zeros(hidden_dim)

    # batch norm
    # AFFINE(N, HID) => BN(N, NHID)
    self.params['gamma3'] = np.ones(hidden_dim)
    self.params['beta3'] = np.zeros(hidden_dim)
    self.bn_params_3 = bn_params.copy()

    # BN(N, NHID) => RELU(N, NHID)

    # AFFINE(N, HID) => OUT(N, CLASSES)    
    # W(CLASSES, HID), B(CLASSES)
    self.params['W4'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b4'] = np.zeros(num_classes)
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)     

 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    gamma1, beta1 = self.params['gamma1'], self.params['beta1']
    gamma2, beta2 = self.params['gamma2'], self.params['beta2']
    gamma3, beta3 = self.params['gamma3'], self.params['beta3']
    conv_params_1 = self.conv_params_1
    conv_params_2 = self.conv_params_2
    bn_params_1 = self.bn_params_1
    bn_params_2 = self.bn_params_2
    bn_params_3 = self.bn_params_3
    pool_params = self.pool_params
    reg = self.reg
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    if y is None:
      bn_params_1['mode'] = 'test'
      bn_params_2['mode'] = 'test'
      bn_params_3['mode'] = 'test'
    else:
      bn_params_1['mode'] = 'train'
      bn_params_2['mode'] = 'train'
      bn_params_3['mode'] = 'train'


    conv_out_1, conv_cache_1 = conv_sbn_relu_forward(
      X, W1, b1, gamma1, beta1,
      conv_params_1, bn_params_1)

    conv_out_2, conv_cache_2 = conv_sbn_relu_pool_forward(
      conv_out_1, W2, b2, gamma2, beta2,
      conv_params_2, bn_params_2, pool_params)

    affine_relu_out, affine_relu_cache = affine_relu_forward(
      conv_out_2, W3, b3, gamma3, beta3, bn_params_3)

    affine, affine_cache = affine_forward(affine_relu_out, W4, b4)

    scores = affine
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    data_loss, dscores = softmax_loss(scores, y)
    reg_loss = (
      0.5 * reg * np.sum(W1 * W1) +
      0.5 * reg * np.sum(W2 * W2) +
      0.5 * reg * np.sum(W3 * W3) +
      0.5 * reg * np.sum(W4 * W4)
    )
    loss = data_loss + reg_loss

    dx, dw, db = affine_backward(dscores, affine_cache)
    grads['W4'] = dw
    grads['b4'] = db

    dx, dw, db, dgamma, dbeta = affine_relu_backward(dx, affine_relu_cache)
    grads['W3'] = dw
    grads['b3'] = db
    grads['gamma3'] = dgamma
    grads['beta3'] = dbeta
    
    dx, dw, db, dgamma, dbeta = conv_sbn_relu_pool_backward(dx, conv_cache_2)
    grads['W2'] = dw
    grads['b2'] = db
    grads['gamma2'] = dgamma
    grads['beta2'] = dbeta
    
    dx, dw, db, dgamma, dbeta = conv_sbn_relu_backward(dx, conv_cache_1)
    grads['W1'] = dw
    grads['b1'] = db
    grads['gamma1'] = dgamma
    grads['beta1'] = dbeta
    
    grads['W1'] += reg * W1
    grads['W2'] += reg * W2
    grads['W3'] += reg * W3
    grads['W4'] += reg * W4

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
