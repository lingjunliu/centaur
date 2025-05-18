
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def batch_norm_inputs():
    list_of_inputs = []

    # Input 1
    input = torch.randn(2, 3, 4, 5).numpy()
    num_features = input.shape[1]
    running_mean = torch.randn(num_features).numpy()
    running_var = torch.rand(num_features).numpy()
    weight = torch.randn(num_features).numpy()
    bias = torch.randn(num_features).numpy()
    training = True
    momentum = 0.1
    eps = 1e-5

    input_dict = {
        "input": input,
        "running_mean": running_mean,
        "running_var": running_var,
        "weight": weight,
        "bias": bias,
        "training": training,
        "momentum": momentum,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input = torch.randn(1, 5, 6, 7).numpy()
    num_features = input.shape[1]
    running_mean = torch.randn(num_features).numpy()
    running_var = torch.rand(num_features).numpy()
    weight = torch.randn(num_features).numpy()
    bias = torch.randn(num_features).numpy()
    training = False
    momentum = 0.2
    eps = 1e-4

    input_dict = {
        "input": input,
        "running_mean": running_mean,
        "running_var": running_var,
        "weight": weight,
        "bias": bias,
        "training": training,
        "momentum": momentum,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input = torch.randn(3, 2, 8, 9).numpy()
    num_features = input.shape[1]
    running_mean = torch.randn(num_features).numpy()
    running_var = torch.rand(num_features).numpy()
    weight = torch.randn(num_features).numpy()
    bias = torch.randn(num_features).numpy()
    training = True
    momentum = 0.05
    eps = 1e-6

    input_dict = {
        "input": input,
        "running_mean": running_mean,
        "running_var": running_var,
        "weight": weight,
        "bias": bias,
        "training": training,
        "momentum": momentum,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input = torch.randn(4, 4, 10, 11).numpy()
    num_features = input.shape[1]
    running_mean = torch.randn(num_features).numpy()
    running_var = torch.rand(num_features).numpy()
    weight = torch.randn(num_features).numpy()
    bias = torch.randn(num_features).numpy()
    training = False
    momentum = 0.15
    eps = 1e-3

    input_dict = {
        "input": input,
        "running_mean": running_mean,
        "running_var": running_var,
        "weight": weight,
        "bias": bias,
        "training": training,
        "momentum": momentum,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input = torch.randn(1, 1, 12, 13).numpy()
    num_features = input.shape[1]
    running_mean = torch.randn(num_features).numpy()
    running_var = torch.rand(num_features).numpy()
    weight = torch.randn(num_features).numpy()
    bias = torch.randn(num_features).numpy()
    training = True
    momentum = 0.3
    eps = 1e-7

    input_dict = {
        "input": input,
        "running_mean": running_mean,
        "running_var": running_var,
        "weight": weight,
        "bias": bias,
        "training": training,
        "momentum": momentum,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = batch_norm_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('batch_norm', list_of_inputs)
