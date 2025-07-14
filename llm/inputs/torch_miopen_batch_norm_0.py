
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def miopen_batch_norm_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.randn(2, 3, 4, 5).astype(np.float32)
    weight = np.random.randn(3).astype(np.float32)
    bias = np.random.randn(3).astype(np.float32)
    running_mean = np.random.randn(3).astype(np.float32)
    running_var = np.random.rand(3).astype(np.float32)
    training = True
    exponential_average_factor = 0.1
    eps = 1e-5

    input_dict = {
        "input": input_tensor,
        "weight": weight,
        "bias": bias,
        "running_mean": running_mean,
        "running_var": running_var,
        "training": training,
        "exponential_average_factor": exponential_average_factor,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.randn(1, 5, 10, 10).astype(np.float32)
    weight = np.random.randn(5).astype(np.float32)
    bias = np.random.randn(5).astype(np.float32)
    running_mean = np.zeros(5).astype(np.float32)
    running_var = np.ones(5).astype(np.float32)
    training = False
    exponential_average_factor = 0.2
    eps = 1e-4

    input_dict = {
        "input": input_tensor,
        "weight": weight,
        "bias": bias,
        "running_mean": running_mean,
        "running_var": running_var,
        "training": training,
        "exponential_average_factor": exponential_average_factor,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.randn(4, 7, 8, 8).astype(np.float32)
    weight = np.random.randn(7).astype(np.float32)
    bias = np.random.randn(7).astype(np.float32)
    running_mean = np.random.randn(7).astype(np.float32)
    running_var = np.abs(np.random.randn(7)).astype(np.float32)  # Ensure variance is positive
    training = True
    exponential_average_factor = 0.05
    eps = 1e-8

    input_dict = {
        "input": input_tensor,
        "weight": weight,
        "bias": bias,
        "running_mean": running_mean,
        "running_var": running_var,
        "training": training,
        "exponential_average_factor": exponential_average_factor,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.random.randn(3, 4, 5, 6).astype(np.float32)
    weight = np.random.randn(4).astype(np.float32)
    bias = np.random.randn(4).astype(np.float32)
    running_mean = np.random.randn(4).astype(np.float32)
    running_var = np.random.rand(4).astype(np.float32)
    training = False
    exponential_average_factor = 0.3
    eps = 1e-6

    input_dict = {
        "input": input_tensor,
        "weight": weight,
        "bias": bias,
        "running_mean": running_mean,
        "running_var": running_var,
        "training": training,
        "exponential_average_factor": exponential_average_factor,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.miopen_batch_norm"] = miopen_batch_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.miopen_batch_norm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.miopen_batch_norm'.")

check_valid('torch.miopen_batch_norm', generated_inputs['torch.miopen_batch_norm'], lib="torch", suffix=0)
