# The Dirichlet Mechanism

[![License](https://img.shields.io/github/license/Donlapark/Dirichlet-Mechanism)](LICENSE)
![Python](https://img.shields.io/badge/python-3.7_|_3.8-blue.svg)
[![CodeQL](https://github.com/donlapark/Dirichlet-Mechanism/actions/workflows/codeql.yml/badge.svg)](https://github.com/donlapark/Dirichlet-Mechanism/actions/workflows/codeql.yml)
![Maintenance](https://img.shields.io/maintenance/yes/2023)

This is the official code associated with the following paper:

> Donlapark Ponnoprat (2022). Dirichlet Mechanism for Differentially Private KL Divergence Minimization. Transactions on Machine Learning Research.

This is a simple implementation of differentially private Naïve Bayes classification and differentially private Bayesian network.
- Flexible options for the underlying private mechanism, namely the Gaussian mechanism, Laplace mechanism, or Dirichlet mechanism.
- Preprocessing pipeline that can take data with non-numeric features and missing values.

## Installation

1. Install the latest version of `numpy`, `scipy`, `pandas` and `sklearn`.
2. Place `models.py`, `samplers.py` and `utils.py` in the working directory.

## Usage

**1. Differentially Private Naïve Bayes Algorithm**

Below is an example of fitting the model and making predictions on the `Adult` dataset. The model is privatized via sampling from the Dirichlet mechanism, where each sampling satisfies $(1,5)$-[Renyi Differential Privacy](https://arxiv.org/abs/1702.07476).
<hr>

```python
from models import DPNaiveBayes
from samplers import (DirichletMechanism,
                      GaussianMechanism,
                      LaplaceMechanism,
                      MLECalculator)
from utils import prepare_labeled_data

seed = 1


# Discretize continuous attributes and split the data, missing values allowed
# The label must be in the last column of the data file.
X_train, X_test, y_train, y_test = prepare_labeled_data("experiments/data/Adult.csv",
                                                        test_size=0.3,
                                                        num_bins=10,
                                                        seed=seed)

# Create a new sampler with (epsilon, lambda)-RDP. Currently, we have:
# 1. DirichletMechanism
# 2. GaussianMechanism
# 3. LaplaceMechanism
DM_sampler = DirichletMechanism(epsilon=1.0, lambda_=5)

# Create a DP naive bayes model with the specified sampler
dpnb = DPNaiveBayes(sampler=DM_sampler, random_state=seed)

# Fit the model
dpnb.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dpnb.predict(X_test)

# Make predicted probabilities on the test set
y_prob = dpnb.predict_proba(X_test)
```
<hr>

**2. Differentially Private Bayesian Network**

Below is an example of fitting the model with a specific graph on the `Adult` dataset. The model is privatized via sampling from the Dirichlet mechanism, where each sampling satisfies $(1,5)$-[Renyi Differential Privacy](https://arxiv.org/abs/1702.07476). 

To compute the log-likelihood on the test data, we have to fit an MLE model (no sampling) on the test set in order to obtain the count data.
<hr>

```python
from models import DPBayesianNetwork
from samplers import (DirichletMechanism,
                      GaussianMechanism,
                      LaplaceMechanism,
                      MLECalculator)
from utils import loglikelihood, prepare_data

seed = 1


# Define a Bayesian networks. The keys are nodes, and the values are associated parents.
adult_net = {"age": [],
             "sex": [],
             "education": ["age"],
             "occupation": ["age", "sex", "education"],
             "capital-gain": ["sex", "education", "occupation"],
             "capital-loss": ["sex", "occupation", "capital-gain"],
             "income": ["occupation", "capital-gain", "capital-loss"]}

# Discretize continuous attributes and split the data, missing values allowed
X_train, X_test = prepare_data("experiments/data/Adult.csv",
                               test_size=0.3,
                               num_bins=10,
                               seed=seed)

# Create a new sampler with (epsilon, lambda)-RDP. Currently, we have:
# 1. DirichletMechanism
# 2. GaussianMechanism
# 3. LaplaceMechanism
DM_sampler = DirichletMechanism(epsilon=1.0, lambda_=5)

# Create a DP Bayesian network with the specified sampler
dpbn = DPBayesianNetwork(bayesian_net=adult_net,
                         sampler=DM_sampler,
                         random_state=seed)

# Fit the model and obtain private parameters
dpbn.fit(X_train)

# To compute the log-likelihood on the test set,
# we need its count data, which can be obtained
# by fitting an MLE model (specifying MLE_calc 
# as the sampler)
MLE_calc = MLECalculator(prior=0.1)
mlebn = DPBayesianNetwork(bayesian_net=adult_net,
                          sampler=MLE_calc,
                          random_state=seed)

# Fit the MLE model and obtain test counts
mlebn.fit(X_test)

# Compute the log-likelihood on the test set
# using the test counts and private parameters
ll = loglikelihood(mlebn.count_dict, dpbn.param_dict)
```

<hr>

