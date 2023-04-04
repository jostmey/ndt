#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2022-11-17
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################

import torchvision
import torch
import pandas as pd

##########################################################################################
# Load data
##########################################################################################

samples_train = torchvision.datasets.MNIST('./', train=True, download=True)
samples_test = torchvision.datasets.MNIST('./', train=False, download=True)

##########################################################################################
# Format data
##########################################################################################

# Format features and labels
#
xs = samples_train.data
num = xs.shape[0]
xs = xs.reshape([ num, -1 ])
xs = xs.type(torch.float32)
ys = samples_train.train_labels

xs_test = samples_test.data
num_test = xs_test.shape[0]
xs_test = xs_test.reshape([ num_test, -1 ])
xs_test = xs_test.type(torch.float32)
ys_test = samples_test.test_labels

# Split into training and validation samples
#
num_train = int(num*5/6)
num_val = num-num_train

js = torch.randperm(num)
js_train = js[:num_train]
js_val = js[num_train:]

xs_train = xs[:num_train]
ys_train = ys[:num_train]

xs_val = xs[num_train:]
ys_val = ys[num_train:]

# Normalizing features
#
mean = torch.mean(xs_train, axis=0, keepdim=True)
variance = torch.var(xs_train, axis=0, keepdim=True)

xs_train = (xs_train-mean)/torch.std(variance+1.0E-8)
xs_val = (xs_val-mean)/torch.std(variance+1.0E-8)
xs_test = (xs_test-mean)/torch.std(variance+1.0E-8)

# Format labels
#
ys_train = ys_train%2
ys_val = ys_val%2
ys_test = ys_test%2

# Convert to dataframes
#
features = []
for i in range(28):
  for j in range(28):
    features.append(str(i+1)+'x'+str(j+1))
labels = ['odd']

xs_train = pd.DataFrame(xs_train.numpy(), columns=features)
ys_train = pd.DataFrame(ys_train.numpy(), columns=labels)

xs_val = pd.DataFrame(xs_val.numpy(), columns=features)
ys_val = pd.DataFrame(ys_val.numpy(), columns=labels)

xs_test = pd.DataFrame(xs_test.numpy(), columns=features)
ys_test = pd.DataFrame(ys_test.numpy(), columns=labels)

##########################################################################################
# Save results
##########################################################################################

xs_train.to_csv('xs_train.csv.gz', index=False, compression='gzip')
ys_train.to_csv('ys_train.csv.gz', index=False, compression='gzip')

xs_val.to_csv('xs_val.csv.gz', index=False, compression='gzip')
ys_val.to_csv('ys_val.csv.gz', index=False, compression='gzip')

xs_test.to_csv('xs_test.csv.gz', index=False, compression='gzip')
ys_test.to_csv('ys_test.csv.gz', index=False, compression='gzip')

