from cs231n.layers import *
from cs231n.fast_layers import *


def affine_relu_forward(x, w, b, gamma, beta, bn_param):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  bn, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
  out, relu_cache = relu_forward(bn)
  cache = (fc_cache, bn_cache, relu_cache)
  return out, cache


def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, bn_cache, relu_cache = cache
  dr = relu_backward(dout, relu_cache)
  dbn, dgamma, dbeta = batchnorm_backward(dr, bn_cache)
  dx, dw, db = affine_backward(dbn, fc_cache)
  return dx, dw, db, dgamma, dbeta


def affine_bn_relu_forward(x, w, b, gamma, beta):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  bn, bn_cache = batchnorm_forward(a, gamma, beta, {'mode': 'train'})
  out, relu_cache = relu_forward(bn)
  cache = (fc_cache, bn_cache, relu_cache)
  return out, cache


def affine_bn_relu_backward(dout, cache, gamma, beta):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, bn_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dbn = batchnorm_backwards(da, bn_cache)
  dx, dw, db = affine_backward(dbn, fc_cache)
  return dx, dw, db



def conv_relu_forward(x, w, b, conv_param):
  """
  A convenience layer that performs a convolution followed by a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  
  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  out, relu_cache = relu_forward(a)
  cache = (conv_cache, relu_cache)
  return out, cache


def conv_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu convenience layer.
  """
  conv_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def conv_sbn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
  """
  A convenience layer that performs a convolution followed by a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  
  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  sbn_out, sbn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
  out, relu_cache = relu_forward(sbn_out)
  cache = (conv_cache, sbn_cache, relu_cache)
  return out, cache


def conv_sbn_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu convenience layer.
  """
  conv_cache, sbn_cache, relu_cache = cache
  dr = relu_backward(dout, relu_cache)
  dsbn, dgamma, dbeta = spatial_batchnorm_backward(dr, sbn_cache)
  dx, dw, db = conv_backward_fast(dsbn, conv_cache)
  return dx, dw, db, dgamma, dbeta


def conv_sbn_relu_pool_forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """

  conv, conv_cache = conv_forward_fast(x, w, b, conv_param)
  sbn, sbn_cache = spatial_batchnorm_forward(conv, gamma, beta, bn_param)
  relu, relu_cache = relu_forward(sbn)
  pool, pool_cache = max_pool_forward_fast(relu, pool_param)
  return pool, (conv_cache, sbn_cache, relu_cache, pool_cache)


def conv_sbn_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, sbn_cache, relu_cache, pool_cache = cache
  dmp = max_pool_backward_fast(dout, pool_cache)
  dr = relu_backward(dmp, relu_cache)
  dsbn, dgamma, dbeta = spatial_batchnorm_backward(dr, sbn_cache)
  dx, dw, db = conv_backward_fast(dsbn, conv_cache)
  return dx, dw, db, dgamma, dbeta

