## Dataset

### Bank Marketing Data Set

Dataset for predicting whether income exceeds $50K/yr based on census data. Also known as "Census Income" dataset. The files `adult.data` and `adult.test` was downloaded from the UC Irvine Machine Learning Repository:

[link](https://archive.ics.uci.edu/ml/datasets/Adult)

To build the dataset, run the following command.

`python3 build_db.py`

The script shuffles the samples, extract the features and labels, split the samples, normalize the features, and save the results.
