## Dataset

### Bank Marketing Data Set

Dataset of images of hand-written digits 0 to 9. The goal is use the image to predict the digit. To make this a binary classification problem, the classification goal has been modified to predict if the digit represented in the image is odd or even.	

To build the dataset, run the following command.

`python3 build_db.py`

The script shuffles the samples, extract the features and labels, split the samples, normalize the features, and save the results.
