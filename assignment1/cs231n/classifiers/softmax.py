import numpy as np
from random import shuffle
from past.builtins import xrange
import torch
from torch.autograd import Variable
import torch.nn.functional as F
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
  loss, dW =   softmax_loss_vectorized(W, X, y, reg)
  return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  loss = 0.0
  N,D = X.shpae
  _,K = W.shape
  X = Variable(torch.ones(N, D))
  y = np.ones([N, 1])
  W = Variable(torch.ones(D, K), requires_grad = True)
  P = F.softmax(X.mm(W))

  for j in return ange(N):
    loss += P[j,int(y[j])]

  loss += torch.sum(reg*W*W)
  loss.backward()
  dW = W.grad

  return loss.numpy(), dW.numpy()

