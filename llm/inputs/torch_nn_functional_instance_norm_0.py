
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def instance_norm_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.rand(2, 3, 4, 5).astype(np.float32)
    running_mean = np.zeros(3).astype(np.float32)
    running_var = np.ones(3).astype(np.float32)
    weight = np.ones(3).astype(np.float32)
    bias = np.zeros(3).astype(np.float32)
    use_input_stats = True
    momentum = 0.1
    eps = 1e-5

    input_dict = {
        "input": input_tensor,
        "running_mean": running_mean,
        "running_var": running_var,
        "weight": weight,
        "bias": bias,
        "use_input_stats": use_input_stats,
        "momentum": momentum,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.rand(1, 4, 3, 2).astype(np.float32)
    running_mean = np.zeros(4).astype(np.float32)
    running_var = np.ones(4).astype(np.float32)
    weight = np.random.rand(4).astype(np.float32)
    bias = np.random.rand(4).astype(np.float32)
    use_input_stats = False
    momentum = 0.2
    eps = 1e-4

    input_dict = {
        "input": input_tensor,
        "running_mean": running_mean,
        "running_var": running_var,
        "weight": weight,
        "bias": bias,
        "use_input_stats": use_input_stats,
        "momentum": momentum,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.rand(3, 2, 2, 2).astype(np.float32)
    running_mean = np.zeros(2).astype(np.float32)
    running_var = np.ones(2).astype(np.float32)
    weight = np.ones(2).astype(np.float32)
    bias = np.zeros(2).astype(np.float32)
    use_input_stats = True
    momentum = 0.9
    eps = 1e-8

    input_dict = {
        "input": input_tensor,
        "running_mean": running_mean,
        "running_var": running_var,
        "weight": weight,
        "bias": bias,
        "use_input_stats": use_input_stats,
        "momentum": momentum,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.random.rand(1, 2, 3).astype(np.float32)
    running_mean = np.zeros(2).astype(np.float32)
    running_var = np.ones(2).astype(np.float32)
    weight = np.ones(2).astype(np.float32)
    bias = np.zeros(2).astype(np.float32)
    use_input_stats = False
    momentum = 0.5
    eps = 1e-6

    input_dict = {
        "input": input_tensor,
        "running_mean": running_mean,
        "running_var": running_var,
        "weight": weight,
        "bias": bias,
        "use_input_stats": use_input_stats,
        "momentum": momentum,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5: 5D input
    input_tensor = np.random.rand(1, 2, 3, 4, 5).astype(np.float32)
    running_mean = np.zeros(2).astype(np.float32)
    running_var = np.ones(2).astype(np.float32)
    weight = np.ones(2).astype(np.float32)
    bias = np.zeros(2).astype(np.float32)
    use_input_stats = True
    momentum = 0.1
    eps = 1e-5

    input_dict = {
        "input": input_tensor,
        "running_mean": running_mean,
        "running_var": running_var,
        "weight": weight,
        "bias": bias,
        "use_input_stats": use_input_stats,
        "momentum": momentum,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Different running mean and var
    input_tensor = np.random.rand(2, 3, 4, 5).astype(np.float32)
    running_mean = np.random.rand(3).astype(np.float32)
    running_var = np.random.rand(3).astype(np.float32) + 0.5
    weight = np.ones(3).astype(np.float32)
    bias = np.zeros(3).astype(np.float32)
    use_input_stats = False
    momentum = 0.1
    eps = 1e-5

    input_dict = {
        "input": input_tensor,
        "running_mean": running_mean,
        "running_var": running_var,
        "weight": weight,
        "bias": bias,
        "use_input_stats": use_input_stats,
        "momentum": momentum,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Small values for eps
    input_tensor = np.random.rand(1, 4, 3, 2).astype(np.float32)
    running_mean = np.zeros(4).astype(np.float32)
    running_var = np.ones(4).astype(np.float32)
    weight = np.random.rand(4).astype(np.float32)
    bias = np.random.rand(4).astype(np.float32)
    use_input_stats = False
    momentum = 0.2
    eps = 1e-9

    input_dict = {
        "input": input_tensor,
        "running_mean": running_mean,
        "running_var": running_var,
        "weight": weight,
        "bias": bias,
        "use_input_stats": use_input_stats,
        "momentum": momentum,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Zero values in the input
    input_tensor = np.zeros((3, 2, 2, 2)).astype(np.float32)
    running_mean = np.zeros(2).astype(np.float32)
    running_var = np.ones(2).astype(np.float32)
    weight = np.ones(2).astype(np.float32)
    bias = np.zeros(2).astype(np.float32)
    use_input_stats = True
    momentum = 0.9
    eps = 1e-8

    input_dict = {
        "input": input_tensor,
        "running_mean": running_mean,
        "running_var": running_var,
        "weight": weight,
        "bias": bias,
        "use_input_stats": use_input_stats,
        "momentum": momentum,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: weights and biases as zeros
    input_tensor = np.random.rand(1, 2, 3).astype(np.float32)
    running_mean = np.zeros(2).astype(np.float32)
    running_var = np.ones(2).astype(np.float32)
    weight = np.zeros(2).astype(np.float32)
    bias = np.zeros(2).astype(np.float32)
    use_input_stats = False
    momentum = 0.5
    eps = 1e-6

    input_dict = {
        "input": input_tensor,
        "running_mean": running_mean,
        "running_var": running_var,
        "weight": weight,
        "bias": bias,
        "use_input_stats": use_input_stats,
        "momentum": momentum,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    
    # Input 10: 3D with different shapes
    input_tensor = np.random.rand(1, 3, 4).astype(np.float32)
    running_mean = np.zeros(3).astype(np.float32)
    running_var = np.ones(3).astype(np.float32)
    weight = np.ones(3).astype(np.float32)
    bias = np.zeros(3).astype(np.float32)
    use_input_stats = True
    momentum = 0.1
    eps = 1e-5

    input_dict = {
        "input": input_tensor,
        "running_mean": running_mean,
        "running_var": running_var,
        "weight": weight,
        "bias": bias,
        "use_input_stats": use_input_stats,
        "momentum": momentum,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.instance_norm"] = instance_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.instance_norm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.instance_norm'.")

check_valid('torch.nn.functional.instance_norm', generated_inputs['torch.nn.functional.instance_norm'], lib="torch", suffix=0)
