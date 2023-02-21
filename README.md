# The Dirichlet Mechanism

This is the official code associated with the following paper:

> Donlapark Ponnoprat (2022). Dirichlet Mechanism for Differentially Private KL Divergence Minimization. Transactions on Machine Learning Research.

The code provided two private models:

**1. Differentially Private Naïve Bayes Algorithm**

Below is an example of fitting the model and making predictions on the `Adult` dataset. The model is privatized via sampling from the Dirichlet mechanism, where each sampling satisfies $(1,5)$-[Renyi Differential Privacy](https://arxiv.org/abs/1702.07476).
<hr>

```
from models import DPBayesianNetwork, DPNaiveBayes
from samplers import (DirichletMechanism,
                      GaussianMechanism,
                      LaplaceMechanism,
                      MLECalculator)
from utils import loglikelihood, prepare_data, prepare_labeled_data

seed = 1

# Discretize continuous attributes and split the data, missing values allowed
X_train, X_test, y_train, y_test = prepare_labeled_data("experiments/data/Adult.csv",
                                                        test_size=0.3,
                                                        num_bins=10,
                                                        seed=seed)

# Create a new sampler with (epsilon, lambda)-RDP. One of 
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

```
# Define a Bayesian networks. The values are parents.
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

# Create a new sampler with (epsilon, lambda)-RDP. One of 
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
dp = DPBayesianNetwork(bayesian_net=adult_net,
                       sampler=MLE_calc,
                       random_state=seed)

# Fit the MLE model and obtain test counts
dp.fit(X_test)

# Compute the log-likelihood on the test set
# using the test counts and private parameters
ll = loglikelihood(dp.count_dict, dpbn.param_dict)
```

<hr>

