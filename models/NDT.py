##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2020-08-13
# Purpose: Implementation neural decision tree (NDT) and variants for PyTorch
# Source: https://gist.github.com/jostmey/f5018f6803df4c6faecd3bca64e5829c
##########################################################################################

import torch

class NDT(torch.nn.Module):
  def __init__(self, num_inputs, tree_depth, num_trees=1, epsilon=1.0e-5, **kwargs):
    super(__class__, self).__init__(**kwargs)
    self.tree_depth = tree_depth
    self.num_trees = num_trees
    self.num_forks = 2**self.tree_depth-1
    self.linear = torch.nn.Linear(num_inputs, self.num_trees*self.num_forks)
    self.norm = torch.nn.BatchNorm1d(self.num_trees*self.num_forks)
    self.sigmoid = torch.nn.Sigmoid()
    self.epsilon = epsilon
  def forward(self, x):
    logits = self.linear(x)
    norms = self.norm(logits)
    reshapes = torch.reshape(norms, [ -1, self.num_trees, self.num_forks ])
    trees_flat = torch.ones_like(reshapes[:,:,0:1])
    j = 0
    for i in range(self.tree_depth):  # Grow the trees
      decisions1 = self.sigmoid(reshapes[:,:,j:j+2**i])
      decisions = torch.stack([ 1.0-decisions1, decisions1 ], axis=3)
      trees = torch.unsqueeze(trees_flat, axis=3)*decisions  # [ batch, tree, decision, 2 ]
      width = int(trees_flat.shape[2])*2
      trees_flat = torch.reshape(trees, [ -1, self.num_trees, width ])
      j += 2**i
    probabilities = torch.sum(trees, axis=2)  # [ batch, tree, 2 ]
    probabilities1 = probabilities[:,:,1:2]  # [ batch, tree, 1 ]
    probability1 = torch.mean(probabilities1, axis=1) # [ batch, 1 ]
    caps = (1.0-2*self.epsilon)*probability1+self.epsilon
    return caps

class NST(torch.nn.Module):
  def __init__(self, num_inputs, tree_depth, num_trees=1, epsilon=1.0e-5, **kwargs):
    super(__class__, self).__init__(**kwargs)
    self.tree_depth = tree_depth
    self.num_trees = num_trees
    self.num_forks = 2**self.tree_depth-1
    self.linear = torch.nn.Linear(num_inputs, self.num_trees*self.num_forks)
    self.norm = torch.nn.BatchNorm1d(self.num_trees*self.num_forks)
    self.sigmoid = torch.nn.Sigmoid()
    self.epsilon = epsilon
  def forward(self, x):
    logits = self.linear(x)
    norms = self.norm(logits)
    reshapes = torch.reshape(norms, [ -1, self.num_trees, self.num_forks ])
    trees_flat = torch.ones_like(reshapes[:,:,0:1])
    j = 0
    for i in range(self.tree_depth):  # Grow the trees
      scale = 1.0/(2**(self.tree_depth-i-1))
      decisions1 = self.sigmoid(scale*reshapes[:,:,j:j+2**i])
      decisions = torch.stack([ 1.0-decisions1, decisions1 ], axis=3)
      trees = torch.unsqueeze(trees_flat, axis=3)*decisions  # [ batch, tree, decision, 2 ]
      width = int(trees_flat.shape[2])*2
      trees_flat = torch.reshape(trees, [ -1, self.num_trees, width ])
      j += 2**i
    probabilities = torch.sum(trees, axis=2)  # [ batch, tree, 2 ]
    probabilities1 = probabilities[:,:,1:2]  # [ batch, tree, 1 ]
    probability1 = torch.mean(probabilities1, axis=1) # [ batch, 1 ]
    caps = (1.0-2*self.epsilon)*probability1+self.epsilon
    return caps

class NGT(torch.nn.Module):
  def __init__(self, num_inputs, tree_depth, num_trees=1, epsilon=1.0e-5, **kwargs):
    super(__class__, self).__init__(**kwargs)
    self.tree_depth = tree_depth
    self.num_trees = num_trees
    self.num_forks = 2**self.tree_depth-1
    self.linear = torch.nn.Linear(num_inputs, self.num_trees*self.num_forks)
    self.norm = torch.nn.BatchNorm1d(self.num_trees*self.num_forks)
    self.sigmoid = torch.nn.Sigmoid()
    self.epsilon = epsilon
  def forward(self, x):
    logits = self.linear(x)
    norms = self.norm(logits)
    reshapes = torch.reshape(norms, [ -1, self.num_trees, self.num_forks ])
    trees_flat = torch.ones_like(reshapes[:,:,0:1])
    j = 0
    for i in range(self.tree_depth):  # Grow the trees
      scale = 1.0/(2**(0.5*(self.tree_depth-i-1)))
      decisions1 = self.sigmoid(scale*reshapes[:,:,j:j+2**i])
      decisions = torch.stack([ 1.0-decisions1, decisions1 ], axis=3)
      trees = torch.unsqueeze(trees_flat, axis=3)*decisions  # [ batch, tree, decision, 2 ]
      width = int(trees_flat.shape[2])*2
      trees_flat = torch.reshape(trees, [ -1, self.num_trees, width ])
      j += 2**i
    probabilities = torch.sum(trees, axis=2)  # [ batch, tree, 2 ]
    probabilities1 = probabilities[:,:,1:2]  # [ batch, tree, 1 ]
    probability1 = torch.mean(probabilities1, axis=1) # [ batch, 1 ]
    caps = (1.0-2*self.epsilon)*probability1+self.epsilon
    return caps

