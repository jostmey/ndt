#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2023-03-28
# Purpose: Train and validate a model
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################

import argparse
import pandas as pd
import torch
import torchmetrics

##########################################################################################
# Arguments
##########################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Base path with input files', type=str, required=True)
parser.add_argument('--output', help='Filename to save the model', type=str, default=None)
parser.add_argument('--batch', help='Batch size', type=int, default=128)
parser.add_argument('--step', help='Step size', type=float, default=0.001)
parser.add_argument('--epochs', help='Number of passes over the dataset', type=int, default=1024)
parser.add_argument('--device', help='Examples are cuda:0 or cpu', type=str, default='cpu')
args = parser.parse_args()

##########################################################################################
# Settings
##########################################################################################

device = torch.device(args.device)

##########################################################################################
# Load data
##########################################################################################

xs_train = pd.read_csv(args.input+'xs_train.csv.gz')
ys_train = pd.read_csv(args.input+'ys_train.csv.gz')

xs_val = pd.read_csv(args.input+'xs_val.csv.gz')
ys_val = pd.read_csv(args.input+'ys_val.csv.gz')

##########################################################################################
# Format data
##########################################################################################

xs_train = torch.tensor(xs_train.to_numpy()).type(torch.float32).to(device)
ys_train = torch.tensor(ys_train.to_numpy()).type(torch.float32).to(device)

xs_val = torch.tensor(xs_val.to_numpy()).type(torch.float32).to(device)
ys_val = torch.tensor(ys_val.to_numpy()).type(torch.float32).to(device)

##########################################################################################
# Data sampling
##########################################################################################

ws_train = 0.5*((ys_train)/torch.sum(ys_train)+(1.0-ys_train)/torch.sum(1.0-ys_train))
ws_train = ws_train.squeeze(1)

ws_val = 0.5*((ys_val)/torch.sum(ys_val)+(1.0-ys_val)/torch.sum(1.0-ys_val))
ws_val = ws_val.squeeze(1)

dataset_train = torch.utils.data.TensorDataset(xs_train, ys_train)
sampler_train = torch.utils.data.WeightedRandomSampler(ws_train, len(dataset_train), replacement=True)
loader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=args.batch, sampler=sampler_train, drop_last=True)

dataset_val = torch.utils.data.TensorDataset(xs_val, ys_val)
sampler_val = torch.utils.data.WeightedRandomSampler(ws_val, len(dataset_val), replacement=True)
loader_val = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=args.batch, sampler=sampler_val, drop_last=True)

##########################################################################################
# Model
##########################################################################################

num_train, num_features = xs_train.shape

class Model(torch.nn.Module):
  def __init__(self, num_inputs):
    super().__init__()
    self.linear = torch.nn.Linear(num_inputs, 1)
    self.norm = torch.nn.BatchNorm1d(1)
    self.act = torch.nn.Sigmoid()
  def forward(self, x):
    l = self.linear(x)
    n = self.norm(l)
    p = self.act(n)
    return p

model = Model(xs_train.shape[1]).to(device)

##########################################################################################
# Loss and optimizer
##########################################################################################

loss = torch.nn.BCELoss()
accuracy = torchmetrics.classification.BinaryAccuracy().to(device)
auroc = torchmetrics.classification.BinaryAUROC().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

##########################################################################################
# Train and validate model
##########################################################################################

i_best = 0
e_best = 1.0e8
state_best = {}

for i in range(args.epochs):

  e_train = 0.0
  model.train()
  for xs_batch, ys_batch in iter(loader_train):
    ps_batch = model(xs_batch)
    e_batch = loss(ps_batch, ys_batch)
    e_train += e_batch/len(loader_train)
    optimizer.zero_grad()
    e_batch.backward()
    optimizer.step()

  e_val = 0.0
  model.eval()
  with torch.no_grad():
    for xs_batch, ys_batch in iter(loader_val):
      ps_batch = model(xs_batch)
      e_batch = loss(ps_batch, ys_batch)
      e_val += e_batch/len(loader_val)
    if e_val < e_best:
      i_best = i
      e_best = e_val
      state_best = model.state_dict()

  print(
    'epoch:', i,
    'e_train:', float(e_train)/0.693,
    'e_val:', float(e_val)/0.693,
    sep='\t', flush=True
  )

with torch.no_grad():
  ps_val = model(xs_val)
  a_val = accuracy(ps_val, ys_val)
  auroc_val = auroc(ps_val, ys_val)
  print(
    'epoch (best):', i_best,
    'e_val (best):', float(e_val)/0.693, 'a_val (best):', 100.0*float(a_val), 'auroc_val (best):', float(auroc_val),
    sep='\t', flush=True
  )

##########################################################################################
# Save model
##########################################################################################

if args.output is not None:
  torch.save(state_best, args.output)
