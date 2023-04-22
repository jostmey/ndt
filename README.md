# Correction the Gradients of Neural Decision Trees (Work in Progress)

## Introduction

Neural decision trees (NDT) aim to combine the interpretability of decision trees with the more accurate but often less interpretable predictions of deep learning models. However, I've noticed that NDTs can have poor predictive accuracy and wanted to understand why. I found that when training NDTs using standard gradient optimization techniques, the gradient tends to vanish at higher-level branches in the tree, leading to suboptimal learning. The problem is similar to but distinct from the vanishing gradient problem, requiring new strategies to correct the issue. As a result, I am working on implementing various strategies to the NDT architecture to address its issues and hopefully improve its performance.

Please refer to the datasets folder for a variety of datasets utilized in this project. Each dataset represents a binary classification problem with tabular data as features.

Explore the models folder to find the diverse models being evaluated. These include fully connected neural networks, transformers, and several NDT models under examination. A hyperparameter search is conducted to assess each model's performance.

Lastly, visit the results folder to view the latest outcomes of our experiments.
