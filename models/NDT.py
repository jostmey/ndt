##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2020-08-13
# Purpose: Implementation neural decision tree (NDT) and variants for PyTorch
# Source: https://gist.github.com/jostmey/f5018f6803df4c6faecd3bca64e5829c
##########################################################################################

import torch

class NDT(torch.nn.Module):
  def __init__(self, depth, num_trees=1, **kwargs):
    super(__class__, self).__init__(**kwargs)
    self.depth = depth
    self.num_trees = num_trees
  def forward(self, x):
    num_forks = 2**self.depth-1
    logits = torch.reshape(x, [ -1, self.num_trees, num_forks ])
    sigmoids = torch.sigmoid(logits)
    trees_flat = torch.ones_like(logits[:,:,0:1])
    j = 0
    for i in range(self.depth):  # Grow the trees
      decisions1 = sigmoids[:,:,j:j+2**i]
      decisions = torch.stack([ 1.0-decisions1, decisions1 ], axis=3)
      trees = torch.unsqueeze(trees_flat, axis=3)*decisions  # [ batch, tree, decision, 2 ]
      width = int(trees_flat.shape[2])*2
      trees_flat = torch.reshape(trees, [ -1, self.num_trees, width ])
      j += 2**i
    probabilities = torch.sum(trees, axis=2)  # [ batch, tree, 2 ]
    probabilities1 = probabilities[:,:,1]  # [ batch, tree ]
    return probabilities1
  @staticmethod
  def num_inputs(depth, num_trees=1):
    return num_trees*(2**depth-1)

class NST(torch.nn.Module):
  def __init__(self, depth, num_trees=1, **kwargs):
    super(__class__, self).__init__(**kwargs)
    self.depth = depth
    self.num_trees = num_trees
  def forward(self, x):
    num_forks = 2**self.depth-1
    logits = torch.reshape(x, [ -1, self.num_trees, num_forks ])
    trees_flat = torch.ones_like(logits[:,:,0:1])
    j = 0
    for i in range(self.depth):  # Grow the trees
      scale = 1.0/(2**(self.depth-i-1))
      decisions1 = torch.sigmoid(scale*logits[:,:,j:j+2**i])
      decisions = torch.stack([ 1.0-decisions1, decisions1 ], axis=3)
      trees = torch.unsqueeze(trees_flat, axis=3)*decisions  # [ batch, tree, decision, 2 ]
      width = int(trees_flat.shape[2])*2
      trees_flat = torch.reshape(trees, [ -1, self.num_trees, width ])
      j += 2**i
    probabilities = torch.sum(trees, axis=2)  # [ batch, tree, 2 ]
    probabilities1 = probabilities[:,:,1]  # [ batch, tree ]
    return probabilities1
  @staticmethod
  def num_inputs(depth, num_trees=1):
    return num_trees*(2**depth-1)

class NCT(torch.nn.Module):
  def __init__(self, depth, num_trees=1, **kwargs):
    super(__class__, self).__init__(**kwargs)
    self.depth = depth
    self.num_trees = num_trees
  def forward(self, x):
    width = 2**(self.depth-1)
    logits = torch.reshape(x, [ -1, self.num_trees, self.depth, width ])  # [ batch, tree, depth, width ]
    sigmoids = torch.sigmoid(logits)
    trees_flat = torch.ones_like(logits[:,:,0:1,0])
    for i in range(self.depth):
      sigmoids_pool = torch.reshape(
        sigmoids[:,:,i,:],
        [ -1, self.num_trees, 2**i, int(width/2**i) ]
      )  # [ batch, tree, fork, pool ]
      decisions1 = torch.mean(sigmoids_pool, axis=3)  # [ batch, tree, fork ]
      decisions = torch.stack([ 1.0-decisions1, decisions1 ], axis=3)  # [ batch, tree, fork, 2 ]
      trees = torch.unsqueeze(trees_flat, axis=3)*decisions  # [ batch, tree, fork, 2 ]
      trees_flat = torch.reshape(
        trees,
        [ -1, self.num_trees, 2*int(trees_flat.shape[2]) ]
      )
    probabilities = torch.sum(trees, axis=2)  # [ batch, tree, 2 ]
    probabilities1 = probabilities[:,:,1]  # [ batch, tree ]
    return probabilities1
  @staticmethod
  def num_inputs(depth, num_trees=1):
    return num_trees*depth*2**(depth-1)

class Trim(torch.nn.Module):
  def __init__(self, epsilon=1.0e-5, **kwargs):
    super(__class__, self).__init__(**kwargs)
    self.epsilon = epsilon
  def forward(self, x):
    return self.epsilon+x*(1.0-2.0*self.epsilon)

class Vote(torch.nn.Module):
  def forward(self, x):
    return torch.mean(x, axis=1, keepdims=True)
