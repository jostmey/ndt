## Dataset

### Bank Marketing Data Set

Dataset of direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit. The file `bank-full.csv` was downloaded from the UC Irvine Machine Learning Repository:

[link](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

To build the dataset, run the following command.

`python3 build_db.py`

The script shuffles the samples, extract the features and labels, split the samples, normalize the features, and save the results.
