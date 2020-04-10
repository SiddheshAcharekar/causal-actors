# Imports

import torch
import torch.nn as nn

import pyro
from pyro.distributions import Bernoulli
from pyro.distributions import Delta
from pyro.distributions import Normal
from pyro.distributions import Uniform
from pyro.distributions import LogNormal
from pyro.infer import SVI
from pyro.infer import Trace_ELBO
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer.autoguide import AutoDiagonalNormal
# initialize the autodiagonal with init_to_feasible instead of init_to_median
from pyro.infer.autoguide import init_to_feasible
import pickle

# Data Loader
import sys

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../box_office/')
from data_loader import load_tensor_data

pyro.set_rng_seed(101)

# Load all data
x_train_tensors, y_train_tensors, actors, full_data = load_tensor_data("../data/ohe_movies.csv")


# Define model

def f_z(params):
    """
    Samples from P(Z) which is the latent factor model's distribution.
    """    
    z_mean0 = params['z_mean0']
    z_std0 = params['z_std0']
    z = pyro.sample("z", Normal(loc = z_mean0, scale = z_std0))
    return z


def f_x(z, params):
    """
    Samples from P(X|Z)
    
    P(X|Z) is a Bernoulli with E(X|Z) = logistic(Z * W),
    where W is a parameter (matrix).  In training the W is
    hyperparameters of the W distribution are estimated such
    that in P(X|Z), the elements of the vector of X are
    conditionally independent of one another given Z.
    """

    def sample_W():
        """
        Sample the W matrix
        
        W is a parameter of P(X|Z) that is sampled from a Normal
        with location and scale hyperparameters w_mean0 and w_std0
        """

        w_mean0 = params['w_mean0']
        w_std0 = params['w_std0']
        W = pyro.sample("W", Normal(loc = w_mean0, scale = w_std0))
        return W
    
    W = sample_W()
    linear_exp = torch.matmul(z, W)
    # sample x using the Bernoulli likelihood
    x = pyro.sample("x", Bernoulli(logits = linear_exp))
    return x

def f_y(x, z, params):
    """
    Samples from P(Y|X, Z)
    
    Y is sampled from a Gaussian where the mean is an
    affine combination of X and Z.  Bayesian linear
    regression is used to estimate the parameters of
    this affine transformation  function.  Use torch.nn.Module to create
    the Bayesian linear regression component of the overall
    model.
    """

    predictors = torch.cat((x, z), 1)

    w = pyro.sample('weight', Normal(params['weight_mean0'], params['weight_std0']))
    b = pyro.sample('bias', Normal(params['bias_mean0'], params['bias_std0']))

    y_hat = (w * predictors).sum(dim=1) + b
    # variance of distribution centered around y
    sigma = pyro.sample('sigma', Normal(params['sigma_mean0'], params['sigma_std0']))
    with pyro.iarange('data', len(predictors)):
        pyro.sample('y', LogNormal(y_hat, sigma))
        return y_hat

def model(params):
    """The full generative causal model"""
    z = f_z(params)
    x = f_x(z, params)
    y = f_y(x, z, params)
    return {'z': z, 'x': x, 'y': y}

# Define guide function for fitting Latent Variable Model
def step_1_guide(params):
    """
    Guide function for fitting P(Z) and P(X|Z) from data
    """
    # Infer z hyperparams
    qz_mean = pyro.param("qz_mean", params['z_mean0'])
    qz_stddv = pyro.param("qz_stddv", params['z_std0'],
                         constraint=constraints.positive)
    
    z = pyro.sample("z", Normal(loc = qz_mean, scale = qz_stddv))
    
    # Infer w params
    qw_mean = pyro.param("qw_mean", params["w_mean0"])
    qw_stddv = pyro.param("qw_stddv", params["w_std0"],
                          constraint=constraints.positive)
    w = pyro.sample("w", Normal(loc = qw_mean, scale = qw_stddv))


# Define guide function for fitting Bayesian Regression Model
def step_2_guide(params):
	"""
	Guide function for fitting P(w|X, Z) i.e regression parameters given observed variables and latent variables.
	"""

    # Z and W are just sampled using param values optimized in previous step
    z = pyro.sample("z", Normal(loc = params['qz_mean'], scale = params['qz_stddv']))
    w = pyro.sample("w", Normal(loc = params['qw_mean'], scale = params['qw_stddv']))
    
    # Infer regression params
    # parameters of (w : weight)
    w_loc = pyro.param('w_loc', params['weight_mean0'])
    w_scale = pyro.param('w_scale', params['weight_std0'])

    # parameters of (b : bias)
    b_loc = pyro.param('b_loc', params['bias_mean0'])
    b_scale = pyro.param('b_scale', params['bias_std0'])
    # parameters of (sigma)
    sigma_loc = pyro.param('sigma_loc', params['sigma_mean0'])
    sigma_scale = pyro.param('sigma_scale', params['sigma_std0'])

    # sample (w, b, sigma)
    w = pyro.sample('weight', Normal(w_loc, w_scale))
    b = pyro.sample('bias', Normal(b_loc, b_scale))
    sigma = pyro.sample('sigma', Normal(sigma_loc, sigma_scale))

# Train the Latent Variable Model
def training_step_1(x_data, params):
	"""
	Function to train latent variable model.
	Conditions on X and fits Z.
	"""
    
    adam_params = {"lr": 0.0005}
    optimizer = Adam(adam_params)

    conditioned_on_x = pyro.condition(model, data = {"x" : x_data})
    svi = SVI(conditioned_on_x, step_1_guide, optimizer, loss=Trace_ELBO())
    
    print("\n Training Z marginal and W parameter marginal...")

    n_steps = 2000
    pyro.set_rng_seed(101)
    # do gradient steps
    pyro.get_param_store().clear()
    for step in range(n_steps):
        loss = svi.step(params)
        if step % 100 == 0:
            print("[iteration %04d] loss: %.4f" % (step + 1, loss/len(x_data)))
            
    # grab the learned variational parameters
    
    updated_params = {k: v for k, v in params.items()}
    for name, value in pyro.get_param_store().items():
        print("Updating value of hypermeter{}".format(name))
        updated_params[name] = value.detach()
        
    return updated_params

# Train the Bayesian Regression
def training_step_2(x_data, y_data, params):
	"""
	Function to train regression model that fits regression parameters to observed X and previously inferred latent Z.
	"""

    print("Training Bayesian regression parameters...")
    pyro.set_rng_seed(101)
    num_iterations = 1000
    pyro.clear_param_store()
    # Create a regression model
    optim = Adam({"lr": 0.003})
    conditioned_on_x_and_y = pyro.condition(model, data = {
        "x": x_data,
        "y" : y_data
    })

    svi = SVI(conditioned_on_x_and_y, step_2_guide, optim, loss=Trace_ELBO(), num_samples=1000)
    for step in range(num_iterations):
        loss = svi.step(params)
        if step % 100 == 0:
            print("[iteration %04d] loss: %.4f" % (step + 1, loss/len(x_data)))
    
    
    updated_params = {k: v for k, v in params.items()}
    for name, value in pyro.get_param_store().items():
        print("Updating value of hypermeter: {}".format(name))
        updated_params[name] = value.detach()
    print("Training complete.")
    return updated_params

# Aggregated function to train entire model
def train_model():
	"""
	Aggregate function to train the entire generative model.
	"""

    num_datapoints, data_dim = x_train_tensors.shape
    
    latent_dim = 30 # can be changed
#     print(torch.zeros(data_dim + latent_dim).shape)
    params0 = {
        'z_mean0': torch.zeros([num_datapoints, latent_dim]),
        'z_std0' : torch.ones([num_datapoints, latent_dim]),
        'w_mean0' : torch.zeros([latent_dim, data_dim]),
        'w_std0' : torch.ones([latent_dim, data_dim]),
        'weight_mean0': torch.zeros(data_dim + latent_dim),
        'weight_std0': torch.ones(data_dim + latent_dim),
        'bias_mean0': torch.tensor(0.),
        'bias_std0': torch.tensor(1.),
        'sigma_mean0' : torch.tensor(1.),
        'sigma_std0' : torch.tensor(0.05)
    }

    params1 = training_step_1(x_train_tensors, params0)
    params2 = training_step_2(x_train_tensors, y_train_tensors, params1)
    return params1, params2

p1, p2 = train_model()

# Save params to disk for inspection
with open('params.pickle', 'wb') as handle:
    pickle.dump(p2, handle, protocol=pickle.HIGHEST_PROTOCOL)