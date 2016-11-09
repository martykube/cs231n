import json
import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)
    self.hidden = hidden_size

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    hidden = X.dot(W1) + b1
    hidden = np.maximum(0, hidden)
    scores = hidden.dot(W2) + b2
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    scaled_scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scaled_scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    correct_logprobs = -np.log(probs[range(N), y])
    data_loss = np.sum(correct_logprobs) / N
    reg_loss = 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2) 
    loss = data_loss + reg_loss

    # print "loss", loss
    # Backward pass: compute gradients

    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    
    dscores = probs
    dscores[range(N), y] -= 1
    dscores /= N
    
    # back prop to W2 and b2
    dW2 = hidden.T.dot(dscores)
    db2 = np.sum(dscores, axis=0)
    dhidden = dscores.dot(W2.T)
    dhidden[hidden <= 0] = 0

    # backprob to W1 and b1
    dW1 = X.T.dot(dhidden)
    db1 = np.sum(dhidden, axis=0)

    dW2 += reg * W2
    dW1 += reg * W1

    grads['W2'] = dW2
    grads['b2'] = db2
    grads['W1'] = dW1
    grads['b1'] = db1

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    self.learning_rate = learning_rate
    self.reg = reg
    self.epocs = num_iters / iterations_per_epoch

    # Use SGD to optimize the parameters in self.model
    self.loss_history = []
    self.w1_update_history = []
    self.train_acc_history = []
    self.val_acc_history = []

    for it in xrange(num_iters):

      # sample for mini batch
      sample_indexes = np.random.choice(num_train, batch_size)
      X_batch = X[sample_indexes]
      y_batch = y[sample_indexes]

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      self.loss_history.append(loss)


      w1_update = -learning_rate * grads['W1']
      param_scale = np.linalg.norm(self.params['W1'].ravel())
      update_scale = np.linalg.norm(w1_update.ravel())
      self.w1_update_history.append(update_scale / param_scale)

      self.params['W1'] += w1_update
      self.params['b1'] += -learning_rate * grads['b1']
      self.params['W2'] += -learning_rate * grads['W2']
      self.params['b2'] += -learning_rate * grads['b2']

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)
        #print "grads W1\n", grads['W1']
        #print "grads b1\n", grads['b1']
        #print "grads W2\n", grads['W2']
        #print "grads b2\n", grads['b2']

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        self.train_acc_history.append(train_acc)
        self.val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': self.loss_history,
      'w1_update_history': self.w1_update_history,
      'train_acc_history': self.train_acc_history,
      'val_acc_history': self.val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """

    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    hidden = X.dot(W1) + b1
    hidden = np.maximum(0, hidden)
    scores = hidden.dot(W2) + b2
    y_pred = np.argmax(scores, axis=1)

    return y_pred


  def write_to_file(self):
    data = {
      'hidden': self.hidden,
      'epocs': self.epocs,
      'learning_rate': self.learning_rate,
      'reg': self.reg,
      'train_acc_history': self.train_acc_history,
      'val_acc_history': self.val_acc_history,
      'loss_history': self.loss_history,
      'w1_update_history': self.w1_update_history,
    }
    
    filename = "e_%d_h_%d_l_%1.1e_r_%1.1e.json" % (self.epocs, self.hidden, self.learning_rate, self.reg)
    print "filename", filename
    with open(filename, 'w') as outfile:
      json.dump(data, outfile, sort_keys=True, indent=4)

