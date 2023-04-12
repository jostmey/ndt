##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2023-04-12
# Purpose: Implementation fully connected neural network models
##########################################################################################

import torch

class BLR(torch.nn.Module):
  def __init__(self, num_inputs, **kwargs):
    super(__class__, self).__init__(**kwargs)
    self.linear = torch.nn.Linear(num_inputs, 1)
    self.norm = torch.nn.BatchNorm1d(1)
    self.act = torch.nn.Sigmoid()
  def forward(self, x):
    logits = self.linear(x)
    norms = self.norm(logits)
    probabilities = self.act(norms)
    return probabilities

class FCNN(torch.nn.Module):
  def __init__(self, num_inputs, num_hidden, dropout=0.0, **kwargs):
    super(__class__, self).__init__(**kwargs)
    self.linear1 = torch.nn.Linear(num_inputs, num_hidden)
    self.norm1 = torch.nn.BatchNorm1d(num_hidden)
    self.act1 = torch.nn.Tanh()
    self.drop = torch.nn.Dropout(dropout)
    self.linear2 = torch.nn.Linear(num_hidden, 1)
    self.norm2 = torch.nn.BatchNorm1d(1)
    self.act2 = torch.nn.Sigmoid()
  def forward(self, x):
    logits1 = self.linear1(x)
    norms1 = self.norm1(logits1)
    hiddens1 = self.act1(norms1)
    drops1 = self.drop(hiddens1)
    logits2 = self.linear2(drops1)
    norms2 = self.norm2(logits2)
    probabilities = self.act2(norms2)
    return probabilities


