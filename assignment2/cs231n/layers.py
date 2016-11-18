import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  x_flat = np.reshape(x, (x.shape[0], -1))
  out = x_flat.dot(w) + b
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################

  dx_flat = dout.dot(w.T)
  dx = np.reshape(dx_flat, x.shape)

  x_flat = np.reshape(x, (x.shape[0], -1))
  dw = x_flat.T.dot(dout)
  db = np.sum(dout, axis=0)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = np.maximum(0, x)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  dx = dout
  dx[ x <= 0] = 0
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    
    mean = np.mean(x, axis=0)
    x_minus_mean = x - mean

    var = np.var(x, axis=0)
    sqrt_var = np.sqrt(var + eps)
    one_over_sqrt_var = 1.0 / sqrt_var

    x_hat = x_minus_mean * one_over_sqrt_var
    out = gamma * x_hat + beta
    running_mean = momentum * running_mean + (1 - momentum) * mean
    running_var = momentum * running_var + (1 - momentum) * var

    cache = (gamma, eps, x_minus_mean, var, sqrt_var, one_over_sqrt_var, x_hat)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################

    x_normalized = (x - running_mean) / np.sqrt(running_var + eps)
    out = gamma * x_normalized + beta

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  (gamma, eps, x_minus_mean, var, sqrt_var, one_over_sqrt_var, xhat) = cache

  N, D = dout.shape

  # 9 + beta
  dbeta = np.sum(dout, axis=0)
  dgamma_xhat = dout

  # 8 xhat * gamma
  dgamma = np.sum(dgamma_xhat * xhat, axis=0)
  dxhat = dgamma_xhat * gamma

  # 7 x_minus_mean * one_over_sqrt_var
  done_over_sqrt_var = np.sum(dxhat * x_minus_mean, axis=0)
  dx_minus_mean_top = dxhat * one_over_sqrt_var

  # 6 1/x
  dsqrt_var = -(sqrt_var ** (-2)) * done_over_sqrt_var

  #5 sqrt(var + eps)
  dvar = 0.5 * ((var + eps)**(-0.5)) * dsqrt_var 

  #4 np.mean
  dsquares = (1.0 / N) * np.ones((N, D)) * dvar 

  # 3 x**2
  dx_minus_mean_bottom = 2.0 * x_minus_mean * dsquares

  # 2 (x - mean)
  # sum inbound gradient
  dx_minus_mean = dx_minus_mean_top + dx_minus_mean_bottom 
  # gradient to input node
  dx_input = dx_minus_mean
  # gradient to mean node
  dmean = -np.sum(dx_minus_mean, axis=0)

  #1 mean(x)
  dx_mean = (1.0 / N) * np.ones((N, D)) * dmean
  
  dx = dx_input + dx_mean
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    mask = (np.random.rand(*x.shape) < (1 - p)) / (1 - p)
    out = x * mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    out = x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    dx = dout * mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx



def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  N, C, HI, WI = x.shape
  F, C, HF, WF = w.shape
  S, P = (conv_param['stride'], conv_param['pad'])
  HP, WP = HI + 2 * P, WI + 2 * P
  HO, WO = ((HP - HF)/S + 1, (WP - WF)/S + 1) 


  xpad = np.pad(x, ((0, 0), (0, 0), (P, P), (P, P)), 'constant')

  xdist = np.empty((N, F, HO, WO, C, HF, WF))
  for f in range(F):
    for i in range(HO):
      for j in range(WO):
        xdist[:, f, i, j] = xpad[:, :, i*S:i*S+HF, j*S:j*S+WF]

  xw = np.empty_like(xdist)
  for f in range(F):
    xw[:, f, :, :] = xdist[:, f, :, :] * w[f]

  xsum = np.sum(xw, axis=(4, 5, 6))

  out = np.empty_like(xsum)
  for n in range(N):
    for f in range(F):
      out[n, f] = xsum[n, f] + b[f]


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, xdist, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives. (N, F, HO, WO)
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b

  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  """
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  (x, xdist, w, b, conv_param) = cache

  N, C, HI, WI = x.shape
  N, F, HO, WO, C, HF, WF = xdist.shape
  S, P = (conv_param['stride'], conv_param['pad'])
  HP, WP = HI + 2 * P, WI + 2 * P
  HO, WO = ((HP - HF)/S + 1, (WP - WF)/S + 1) 

  # back prop to dxsum and db
  dxsum = dout
  db = np.sum(dout, axis=(0, 2, 3))

  # back prop sum_filter to dxw
  dxw = np.empty_like(xdist)
  for n in range(N):
    for f in range(F):
      for i in range(HO):
        for j in range(WO):
          dxw[n, f, i, j] = np.ones((C, HF, WF)) * dxsum[n, f, i, j]


  # back prop mutliply_w to dxdist and dw
  dw = np.sum(dxw * xdist, axis=(0, 2, 3))
  dxdist = np.empty_like(xdist)
  for i in range(HO):
    for j in range(WO):
      dxdist[:, :, i, j] = dxw[:, :, i, j] * w
        
        
  # back prop dist to xpad
  dxpad = np.zeros((N, C, HP, WP))
  for n in range(N):
    for f in range(F):
      for y_out in range(HO):
        for x_out in range (WO):
          # get the upper left filter input coordinates
          (y_in_ul, x_in_ul) = (y_out * S, x_out * S)
          for y_filter in range(HF):
            for x_filter in range(WF):
              (y_in, x_in) = (y_in_ul + y_filter, x_in_ul + x_filter)
              #for c in range(C):
              dxpad[n, :, y_in, x_in] += dxdist[n, f, y_out, x_out, :, y_filter, x_filter]

                        
  # back prop pad
  dx = dxpad[:, :, 1:HI + 2 * P - 1, 1:WI + 2 * P - 1]

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  N, C, H, W = x.shape
  HP, WP, S = (pool_param['pool_height'], pool_param['pool_width'], pool_param['stride'])

  H_steps = (H - HP) / S + 1
  W_steps = (W - WP) / S + 1

  out = np.empty((N, C, H_steps, W_steps), dtype=x.dtype)
  activations = np.empty((N, C, H_steps, W_steps, 2), dtype=np.uint)

  for row in range(H_steps):
    for col in range(W_steps):

        start_row, start_col = row * S, col * S
        x_to_pool = x[:, :, start_row: start_row + HP, start_col: start_col + WP]
  
        max_width = np.max(x_to_pool, axis=3)
        args_width = np.argmax(x_to_pool, axis=3)
        max_height = np.max(max_width, axis=2)
        args_height = np.argmax(max_width, axis=2)
        
        out[:, :, row, col] = max_height

        for p in range(N):
          for c in range(C):
            active_row = args_height[p, c]
            active_col = args_width[p, c, active_row]
            activations[p, c, row, col] = np.array([active_row, active_col])

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param, activations)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  (x, pool_param, activations) = cache
  
  N, C, H, W = x.shape
  N, C, HS, WS = dout.shape
  HP, WP, S = (pool_param['pool_height'], pool_param['pool_width'], pool_param['stride'])
  
  dx = np.zeros_like(x)
  for n in range(N):
    for c in range(C):
      for row in range(HS):
        for col in range(WS):
          start_row, start_col = row * S, col * S
          target_row, target_col = np.array([start_row, start_col]) + \
                                   activations[n, c, row, col]
          dx[n, c, target_row, target_col] = dout[n, c, row, col]
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  N, C, H, W = x.shape
  x = np.transpose(x, (0, 2, 3, 1))
  x = x.reshape((N * H * W, C))
  out, cache = batchnorm_forward(x, gamma, beta, bn_param)
  out = out.reshape((N, H, W, C))
  out = np.transpose(out, (0, 3, 1, 2))
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  N, C, H, W = dout.shape
  dout = np.transpose(dout, (0, 2, 3, 1)).reshape((N * H * W, C))
  dx, dgamma, dbeta = batchnorm_backward(dout, cache)
  dx = np.transpose(dx.reshape((N, H, W, C)), (0, 3, 1, 2))

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
