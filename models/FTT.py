##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2023-03-28
# Purpose: Transformer model for numeric, tabular data
# Install: pip3 install tab-transformer-pytorch
# Source: https://github.com/lucidrains/tab-transformer-pytorch
##########################################################################################

import torch
from tab_transformer_pytorch import FTTransformer

class FTT(torch.nn.Module):
  def __init__(self, num_inputs, attn_dropout, ff_dropout):
    super().__init__()
    self.trans = FTTransformer(
      categories=(), num_continuous=num_inputs,
      dim=32, dim_out=1,
      depth=6, heads=8,
      attn_dropout=attn_dropout, ff_dropout=ff_dropout
    )
    self.norm = torch.nn.BatchNorm1d(1)
    self.act = torch.nn.Sigmoid()
  def forward(self, x):
    dummies = torch.randint(0, 5, (x.shape[0], 0)) # Dummy category input
    logits = self.trans(dummies, x)
    norms = self.norm(logits)
    probabilities = self.act(norms)
    return probabilities
