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
               dtype=np.float32, num_conv_layers=1):
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
    self.num_conv_layers = num_conv_layers
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

    self.conv_params = {}
    self.pool_params = {}
    self.bn_params = {}

    # input channels, height, width
    C, H, W = input_dim
    F = num_filters

    # # One or more layers
    F_IN = C
    F_OUT = F
    H_CURR = H
    W_CURR = W
    for layer in range(self.num_conv_layers):
      
      # first conv => sbn => relu
 
     # IN(N, C, H, W) => CONV(N, F, H, W)
      # Use padding at (filter_size -1) / 2 and output dimensions are conserved...
      self.params['W_%d_1' % layer] = weight_scale * \
                          np.random.randn(F_OUT, F_IN, filter_size, filter_size)
      self.params['b_%d_1' % layer] = np.zeros(F_OUT)
      self.conv_params['%d_1' % layer] = conv_params.copy()
      # first layer C is Channels, after that is Filters

      # Spatial batch normalization
      # CONV(N, F, H, W) =>  SBN(N, F, H, W)
      self.params['gamma_%d_1' % layer] = np.ones(F_OUT)
      self.params['beta_%d_1' % layer] = np.zeros(F_OUT)
      self.bn_params['%d_1'  % layer] = bn_params.copy()

      # # SBN(N, F, H, W) => RELU(N, F, H, W)
    
      # # second conv => sbn => relu
 
      # # IN(N, F, H, W) => CONV(N, F, H, W)
      # # Use padding at (filter_size -1) / 2 and output dimensions are conserved...
      self.params['W_%d_2' % layer] = weight_scale * \
                                      np.random.randn(F_OUT, F_OUT, filter_size, filter_size)
      self.params['b_%d_2' % layer] = np.zeros(F_OUT)
      self.conv_params['%d_2' % layer] = conv_params.copy()

      # Spatial batch normalization
      # CONV(N, F, H, W) =>  SBN(N, F, H, W)
      self.params['gamma_%d_2' % layer] = np.ones(F_OUT)
      self.params['beta_%d_2' % layer] = np.zeros(F_OUT)
      self.bn_params['%d_2'  % layer] = bn_params.copy()

      # SBN(N, F, H, W) => RELU(N, F, H, W)
    
      # maxpool
      #
      # RELU(N, F, H, W) => MAXPOOL(N, F, HP, WP)
      # pool_params = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
      # (H - F) / S + 1
      H_CURR = (H_CURR - 2) / 2 + 1
      W_CURR = (W_CURR - 2) / 2 + 1
      self.pool_params['%d' % layer] = pool_params.copy()
      
      # Update for next layer
      F_IN = F_OUT
      
    # MAXPOOL(N, F, HP, WP) => AFFINE(N, NHID)
    # W(F * HP * WP, NHID), seems backwards, I know...
    # affine will do the flattening
    layer = self.num_conv_layers
    self.params['W_%d'  % layer] = weight_scale * np.random.randn(
      F * H_CURR * W_CURR, hidden_dim)
    self.params['b_%d' % layer] = np.zeros(hidden_dim)
    self.params['gamma_%d' % layer] = np.ones(hidden_dim)
    self.params['beta_%d' % layer] = np.zeros(hidden_dim)
    self.bn_params['_%d' % layer] = bn_params.copy()

    # AFFINE(N, HID) => AFFINE(N, CLASSES)    
    # W(CLASSES, HID), B(CLASSES)
    layer += 1
    self.params['W_%d' % layer] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b_%d' % layer] = np.zeros(num_classes)
    self.params['gamma_%d' % layer] = np.ones(num_classes)
    self.params['beta_%d' % layer] = np.zeros(num_classes)
    self.bn_params['_%d' % layer] = bn_params.copy()
    
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
    reg = self.reg
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    if y is None:
      for k, v in self.bn_params.iteritems():
        self.bn_params[k]['mode'] = 'test'
    else:
      for k, v in self.bn_params.iteritems():
        self.bn_params[k]['mode'] = 'train'

    conv_cache = {}
    for layer in range(self.num_conv_layers):
      conv_out_1, conv_cache['%d_1' % layer] = conv_sbn_relu_forward(
        X, 
        self.params['W_%d_1' % layer], 
        self.params['b_%d_1' % layer], 
        self.params['gamma_%d_1' % layer], 
        self.params['beta_%d_1' % layer],
        self.conv_params['%d_1' % layer], 
        self.bn_params['%d_1' % layer])

      conv_out_2, conv_cache['%d_2' % layer] = conv_sbn_relu_pool_forward(
        conv_out_1,
        self.params['W_%d_2' % layer], 
        self.params['b_%d_2' % layer], 
        self.params['gamma_%d_2' % layer], 
        self.params['beta_%d_2' % layer],
        self.conv_params['%d_2' % layer], 
        self.bn_params['%d_2' % layer],
        self.pool_params['%d' % layer])

      X = conv_out_2

    layer = self.num_conv_layers
    affine_relu_1, affine_relu_cache_1 = affine_relu_forward(
      conv_out_2,
      self.params['W_%d' % layer], 
      self.params['b_%d' % layer], 
      self.params['gamma_%d' % layer], 
      self.params['beta_%d' % layer],
      self.bn_params['_%d' % layer])

    layer += 1
    affine_relu_2, affine_relu_cache_2 = affine_relu_forward(
      affine_relu_1,
      self.params['W_%d' % layer], 
      self.params['b_%d' % layer], 
      self.params['gamma_%d' % layer], 
      self.params['beta_%d' % layer],
      self.bn_params['_%d' % layer])

    # scores = affine
    scores = affine_relu_2
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
    
    
#    orig_shape = scores.shape
#    print "orig_shape", orig_shape

#    scores = scores.reshape((orig_shape[0], -1))
#    print "scores.shape", scores.shape
#    print "y.shape", y.shape

    data_loss, dscores = softmax_loss(scores, y)

#    dscores = dscores.reshape(orig_shape)
#    print dscores.shape

    reg_loss = 0
    for layer in range(self.num_conv_layers):
      w1 = self.params['W_%d_1' % layer]
      reg_loss += np.sum(w1 * w1) 
      w2 = self.params['W_%d_2' % layer]
      reg_loss += np.sum(w2 * w2) 
    layer = self.num_conv_layers
    reg_loss += np.sum(self.params['W_%d' % layer] * self.params['W_%d' % layer])
    layer += 1
    reg_loss += np.sum(self.params['W_%d' % layer] * self.params['W_%d' % layer])
    reg_loss *= 0.5 * reg

    loss = data_loss + reg_loss

    dx = dscores

    layer = self.num_conv_layers + 1
    dx, dw, db, dgamma, dbeta = affine_relu_backward(dscores, affine_relu_cache_2)
    grads['W_%d' % layer] = dw
    grads['b_%d' % layer] = db
    grads['gamma_%d' % layer] = dgamma
    grads['beta_%d' % layer] = dbeta

    layer -= 1
    dx, dw, db, dgamma, dbeta = affine_relu_backward(dx, affine_relu_cache_1)
    grads['W_%d' % layer] = dw
    grads['b_%d' % layer] = db
    grads['gamma_%d' % layer] = dgamma
    grads['beta_%d' % layer] = dbeta

    for layer in reversed(range(self.num_conv_layers)):
    
      dx, dw, db, dgamma, dbeta = conv_sbn_relu_pool_backward(dx, conv_cache['%d_2' % layer])
      grads['W_%d_2' % layer] = dw
      grads['b_%d_2' % layer] = db
      grads['gamma_%d_2' % layer] = dgamma
      grads['beta_%d_2' % layer] = dbeta
    
      dx, dw, db, dgamma, dbeta = conv_sbn_relu_backward(dx, conv_cache['%d_1' % layer])
      grads['W_%d_1' % layer] = dw
      grads['b_%d_1' % layer] = db
      grads['gamma_%d_1' % layer] = dgamma
      grads['beta_%d_1' % layer] = dbeta

      
    # regularization contribution to gradient
    for layer in range(self.num_conv_layers):
      name = 'W_%d_1' % layer
      grads[name] += reg * self.params[name]
      name = 'W_%d_2' % layer
      grads[name] += reg * self.params[name]

    layer = self.num_conv_layers + 1
    grads['W_%d' % layer] += reg * self.params['W_%d' % layer]
    layer -= 1
    grads['W_%d' % layer] += reg * self.params['W_%d' % layer]

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
