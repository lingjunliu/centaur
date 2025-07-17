
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def miopen_batch_norm_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.rand(2, 3, 4, 5).astype(np.float32)
    weight_tensor = np.random.rand(3).astype(np.float32)
    bias_tensor = np.random.rand(3).astype(np.float32)
    running_mean_tensor = np.random.rand(3).astype(np.float32)
    running_var_tensor = np.random.rand(3).astype(np.float32)
    training_flag = True
    exponential_average_factor_val = 0.1
    eps_val = 1e-5

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "running_mean": running_mean_tensor,
        "running_var": running_var_tensor,
        "training": training_flag,
        "exponential_average_factor": exponential_average_factor_val,
        "eps": eps_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.rand(1, 5, 10, 10).astype(np.float32)
    weight_tensor = np.random.rand(5).astype(np.float32)
    bias_tensor = np.random.rand(5).astype(np.float32)
    running_mean_tensor = np.random.rand(5).astype(np.float32)
    running_var_tensor = np.random.rand(5).astype(np.float32)
    training_flag = False
    exponential_average_factor_val = 0.5
    eps_val = 1e-8

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "running_mean": running_mean_tensor,
        "running_var": running_var_tensor,
        "training": training_flag,
        "exponential_average_factor": exponential_average_factor_val,
        "eps": eps_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.rand(4, 8, 2, 2).astype(np.float32)
    weight_tensor = np.random.rand(8).astype(np.float32)
    bias_tensor = np.random.rand(8).astype(np.float32)
    running_mean_tensor = np.random.rand(8).astype(np.float32)
    running_var_tensor = np.random.rand(8).astype(np.float32)
    training_flag = True
    exponential_average_factor_val = 0.9
    eps_val = 1e-3

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "running_mean": running_mean_tensor,
        "running_var": running_var_tensor,
        "training": training_flag,
        "exponential_average_factor": exponential_average_factor_val,
        "eps": eps_val
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
