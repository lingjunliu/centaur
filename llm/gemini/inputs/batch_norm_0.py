
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def batch_norm_inputs():
    list_of_inputs = []

    # Input 1: Basic case with 2D input
    input = np.random.randn(2, 3).astype(np.float32)
    running_mean = np.random.randn(3).astype(np.float32)
    running_var = np.abs(np.random.randn(3)).astype(np.float32)
    weight = np.random.randn(3).astype(np.float32)
    bias = np.random.randn(3).astype(np.float32)
    input_dict = {
        "input": input,
        "running_mean": running_mean,
        "running_var": running_var,
        "weight": weight,
        "bias": bias,
        "training": False,
        "momentum": 0.1,
        "eps": 1e-5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 4D input (batch, channel, height, width)
    input = np.random.randn(4, 3, 10, 10).astype(np.float32)
    running_mean = np.random.randn(3).astype(np.float32)
    running_var = np.abs(np.random.randn(3)).astype(np.float32)
    weight = np.random.randn(3).astype(np.float32)
    bias = np.random.randn(3).astype(np.float32)
    input_dict = {
        "input": input,
        "running_mean": running_mean,
        "running_var": running_var,
        "weight": weight,
        "bias": bias,
        "training": True,
        "momentum": 0.1,
        "eps": 1e-5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D input (batch, channel, length)
    input = np.random.randn(2, 5, 20).astype(np.float32)
    running_mean = np.random.randn(5).astype(np.float32)
    running_var = np.abs(np.random.randn(5)).astype(np.float32)
    weight = np.random.randn(5).astype(np.float32)
    bias = np.random.randn(5).astype(np.float32)
    input_dict = {
        "input": input,
        "running_mean": running_mean,
        "running_var": running_var,
        "weight": weight,
        "bias": bias,
        "training": False,
        "momentum": 0.1,
        "eps": 1e-5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Different eps and momentum
    input = np.random.randn(3, 4, 5, 5).astype(np.float32)
    running_mean = np.random.randn(4).astype(np.float32)
    running_var = np.abs(np.random.randn(4)).astype(np.float32)
    weight = np.random.randn(4).astype(np.float32)
    bias = np.random.randn(4).astype(np.float32)
    input_dict = {
        "input": input,
        "running_mean": running_mean,
        "running_var": running_var,
        "weight": weight,
        "bias": bias,
        "training": True,
        "momentum": 0.5,
        "eps": 1e-8
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Input with negative values
    input = np.random.randn(1, 2, 3, 3).astype(np.float32) - 2
    running_mean = np.random.randn(2).astype(np.float32)
    running_var = np.abs(np.random.randn(2)).astype(np.float32)
    weight = np.random.randn(2).astype(np.float32)
    bias = np.random.randn(2).astype(np.float32)
    input_dict = {
        "input": input,
        "running_mean": running_mean,
        "running_var": running_var,
        "weight": weight,
        "bias": bias,
        "training": False,
        "momentum": 0.1,
        "eps": 1e-5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 5D input
    input = np.random.randn(2, 3, 4, 5, 6).astype(np.float32)
    running_mean = np.random.randn(3).astype(np.float32)
    running_var = np.abs(np.random.randn(3)).astype(np.float32)
    weight = np.random.randn(3).astype(np.float32)
    bias = np.random.randn(3).astype(np.float32)
    input_dict = {
        "input": input,
        "running_mean": running_mean,
        "running_var": running_var,
        "weight": weight,
        "bias": bias,
        "training": True,
        "momentum": 0.1,
        "eps": 1e-5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.batch_norm"] = batch_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.batch_norm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.batch_norm'.")

check_valid('torch.nn.functional.batch_norm', generated_inputs['torch.nn.functional.batch_norm'], lib="torch")
