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

names = [
  'age', 'workclass', 'fnlwgt', 'education', 'education-num',
  'marital-status', 'occupation', 'relationship', 'race', 'sex',
  'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

df = pd.read_csv('adult.data', names=names, header=None)
df_test = pd.read_csv('adult.test', names=names, header=None, skiprows=1)

##########################################################################################
# Shuffle data
##########################################################################################

df = df.sample(frac=1, random_state=809167)

##########################################################################################
# Merge samples
##########################################################################################

df_concat = pd.concat([ df, df_test ], axis=0)

##########################################################################################
# Convert data
##########################################################################################

# Remove 'Holand-Netherlands', which is not frequent enough to appear in the training dataset
#
df_concat['native-country'] = df_concat['native-country'].replace('Holand-Netherlands', '?')
df_concat['native-country'] = df_concat['native-country'].replace(' Holand-Netherlands', ' ?')

# Convert categorical variables to numerical one-hot encoded variables
#
for name in [ 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country' ]:
  df_concat[name] = df_concat[name].str.strip()
  df_onehot = pd.get_dummies(df_concat[name], prefix=name)
  df_drop = df_concat.drop(columns=[ name ])
  df_concat = pd.concat([ df_drop, df_onehot ], axis=1)

#
#
df_concat['income'] = df_concat['income'].replace(' <=50K.', ' <=50K')
df_concat['income'] = df_concat['income'].replace(' >50K.', ' >50K')

# Convert binary variables to numerical variables
#
for name, one in [ ( 'sex', 'Male' ), ( 'income', '>50K' ) ]:
  df_concat[name] = df_concat[name].str.strip()
  df_binary = df_concat[name].eq(one).astype('float')
  df_drop = df_concat.drop(columns=[ name ])
  df_concat = pd.concat([ df_drop, df_binary ], axis=1)

##########################################################################################
# Separate features and labels
##########################################################################################

ys = df_concat['income']
xs = df_concat.drop(columns=[ 'income' ])

##########################################################################################
# Split data
##########################################################################################

num_test = df_test.shape[0]

xs_train = xs.iloc[:-2*num_test]
ys_train = ys.iloc[:-2*num_test]

xs_val = xs.iloc[-2*num_test:-num_test]
ys_val = ys.iloc[-2*num_test:-num_test]

xs_test = xs.iloc[-num_test:]
ys_test = ys.iloc[-num_test:]

##########################################################################################
# Normalize features
##########################################################################################

means = xs_train.mean()
stds = xs_train.std()

xs_train = (xs_train-means)/stds
xs_val = (xs_val-means)/stds
xs_test = (xs_test-means)/stds

##########################################################################################
# Save results
##########################################################################################

xs_train.to_csv('xs_train.csv.gz', index=False, compression='gzip')
ys_train.to_csv('ys_train.csv.gz', index=False, compression='gzip')

xs_val.to_csv('xs_val.csv.gz', index=False, compression='gzip')
ys_val.to_csv('ys_val.csv.gz', index=False, compression='gzip')

xs_test.to_csv('xs_test.csv.gz', index=False, compression='gzip')
ys_test.to_csv('ys_test.csv.gz', index=False, compression='gzip')

