#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2023-03-28
# Purpose: Train, validate, and test a model
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################

import argparse
import pandas as pd
import torch
import torchmetrics
from FCNN import *

##########################################################################################
# Arguments
##########################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='Path to folder holding dataset files', type=str, required=True)
parser.add_argument('--output', help='Filename to save the model', type=str, default=None)
parser.add_argument('--batch', help='Batch size', type=int, default=1024)
parser.add_argument('--step', help='Step size', type=float, default=0.01)
parser.add_argument('--epochs', help='Number of passes over the dataset', type=int, default=128)
parser.add_argument('--device', help='Examples are cuda:0 or cpu', type=str, default='cpu')
args = parser.parse_args()

##########################################################################################
# Settings
##########################################################################################

device = torch.device(args.device)

##########################################################################################
# Load data
##########################################################################################

xs_train = pd.read_csv(args.dataset+'xs_train.csv.gz')
ys_train = pd.read_csv(args.dataset+'ys_train.csv.gz')

xs_val = pd.read_csv(args.dataset+'xs_val.csv.gz')
ys_val = pd.read_csv(args.dataset+'ys_val.csv.gz')

xs_test = pd.read_csv(args.dataset+'xs_test.csv.gz')
ys_test = pd.read_csv(args.dataset+'ys_test.csv.gz')

##########################################################################################
# Format data
##########################################################################################

xs_train = torch.tensor(xs_train.to_numpy()).type(torch.float32).to(device)
ys_train = torch.tensor(ys_train.to_numpy()).type(torch.float32).to(device)

xs_val = torch.tensor(xs_val.to_numpy()).type(torch.float32).to(device)
ys_val = torch.tensor(ys_val.to_numpy()).type(torch.float32).to(device)

xs_test = torch.tensor(xs_test.to_numpy()).type(torch.float32).to(device)
ys_test = torch.tensor(ys_test.to_numpy()).type(torch.float32).to(device)

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
# Metrics
##########################################################################################

loss = torch.nn.BCELoss()
accuracy = torchmetrics.classification.BinaryAccuracy().to(device)
auroc = torchmetrics.classification.BinaryAUROC().to(device)

##########################################################################################
# Modelling
##########################################################################################

# For holding the best model
#
i_best = -1
e_best = 1.0e8
hyper_best = {}
state_best = {}

# Loop over hyperparameters shared by all models
#
for lr in [ 0.1, 0.03, 0.01, 0.003, 0.001 ]:

  # Record hyperparameters
  #
  hyper = {
   'lr': lr
  }

  # Create model
  #
  model = BLR(num_inputs=xs_train.shape[1])
  model = model.to(device)

  # For holding a bbetterest model
  #
  i_better = -1
  e_better = 1.0e8
  hyper_better = {}
  state_better = {}

  # Re-initialize the optimizer
  #
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  # Print report
  #
  print('[NEW MODEL]', sep='\t', flush=True)

  # Train and validate the model
  #
  for i in range(args.epochs):

    # Traing model for one epoch
    #
    e_train = 0.0
    model.train()
    for xs_batch, ys_batch in iter(loader_train):
      ps_batch = model(xs_batch)
      e_batch = loss(ps_batch, ys_batch)
      e_train += e_batch/len(loader_train)
      optimizer.zero_grad()
      e_batch.backward()
      optimizer.step()

    # Validate model
    #
    e_val = 0.0
    model.eval()
    with torch.no_grad():
      for xs_batch, ys_batch in iter(loader_val):
        ps_batch = model(xs_batch)
        e_batch = loss(ps_batch, ys_batch)
        e_val += e_batch/len(loader_val)
      if e_val < e_better:
        i_better = i
         e_better = e_val
         hyper_better = hyper
        state_better = model.state_dict()
      if e_val < e_best:
        i_best = i
        e_best = e_val
        hyper_best = hyper
        state_best = model.state_dict()

    # Print report
    #
    print(
      '[TRAIN]',
      'i:', i,
      'e_train:', float(e_train)/0.693,
      'e_val:', float(e_val)/0.693,
      sep='\t', flush=True
    )

  # Print report with early stopping
  #
  with torch.no_grad():
    model.load_state_dict(state_better)
    model.eval()
    ps_val = model(xs_val)
    print(
      '[VALIDATE]',
      'i_better:', i_better,
      'a_val:', 100.0*float(accuracy(ps_val, ys_val)),
      'auroc_val:', float(auroc(ps_val, ys_val)),
      'hyper_better:', hyper_better,
      sep='\t', flush=True
    )

# Print report from hyperparameter search
#
with torch.no_grad():
  model = BLR(num_inputs=xs_train.shape[1])
  model.load_state_dict(state_best)
  model.eval()
  ps_val = model(xs_val)
  ps_test = model(xs_test)
  print(
    '[TEST]',
    'i_best:', i_best,
    'a_val:', 100.0*float(accuracy(ps_val, ys_val)),
    'auroc_val:', float(auroc(ps_val, ys_val)),
    'a_test:', 100.0*float(accuracy(ps_test, ys_test)),
    'auroc_test:', float(auroc(ps_test, ys_test)),
    'hyper_best:', hyper_best,
    sep='\t', flush=True
  )

##########################################################################################
# Save model
##########################################################################################

if args.output is not None:
  torch.save(state_best, args.output)
