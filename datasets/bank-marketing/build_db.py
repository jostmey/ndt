#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2023-03-28
# Purpose: Build database
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################

import pandas as pd

##########################################################################################
# Load data
##########################################################################################

df = pd.read_csv('bank-full.csv', delimiter=';')

##########################################################################################
# Convert data
##########################################################################################

# Convert categorical variables to numerical one-hot encoded variables
#
for name in [ 'job', 'marital', 'education', 'contact', 'month', 'poutcome' ]:
  df_onehot = pd.get_dummies(df[name], prefix=name)
  df_drop = df.drop(columns=[ name ])
  df = pd.concat([ df_drop, df_onehot ], axis=1)

# Convert binary variables to numerical variables
#
for name in [ 'default', 'housing', 'loan', 'y' ]:
  df_binary = df[name].eq('yes').astype('float')
  df_drop = df.drop(columns=[ name ])
  df = pd.concat([ df_drop, df_binary ], axis=1)

##########################################################################################
# Shuffle data
##########################################################################################

df = df.sample(frac=1, random_state=809167)

##########################################################################################
# Separate features and labels
##########################################################################################

ys = df['y']
xs = df.drop(columns=[ 'y' ])

##########################################################################################
# Split data
##########################################################################################

num = df.shape[0]

xs_train = xs.iloc[:int(0.6*num)]
ys_train = ys.iloc[:int(0.6*num)]

xs_val = xs.iloc[int(0.6*num):int(0.8*num)]
ys_val = ys.iloc[int(0.6*num):int(0.8*num)]

xs_test = xs.iloc[int(0.8*num):]
ys_test = ys.iloc[int(0.8*num):]

##########################################################################################
# Normalize features
##########################################################################################

means = xs_train.mean()
stds = xs_train.std()

xs = (xs-means)/stds
xs_train = (xs_train-means)/stds
xs_val = (xs_val-means)/stds
xs_test = (xs_test-means)/stds

##########################################################################################
# Save results
##########################################################################################

xs.to_csv('xs.csv.gz', index=False, compression='gzip')
ys.to_csv('ys.csv.gz', index=False, compression='gzip')

xs_train.to_csv('xs_train.csv.gz', index=False, compression='gzip')
ys_train.to_csv('ys_train.csv.gz', index=False, compression='gzip')

xs_val.to_csv('xs_val.csv.gz', index=False, compression='gzip')
ys_val.to_csv('ys_val.csv.gz', index=False, compression='gzip')

xs_test.to_csv('xs_test.csv.gz', index=False, compression='gzip')
ys_test.to_csv('ys_test.csv.gz', index=False, compression='gzip')

