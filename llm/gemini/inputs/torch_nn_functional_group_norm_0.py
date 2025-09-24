
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def group_norm_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.randn(2, 6, 4, 4).astype(np.float32)
    num_groups = 2
    weight = np.random.randn(6).astype(np.float32)
    bias = np.random.randn(6).astype(np.float32)
    eps = 1e-5

    input_dict = {
        "input": input_tensor,
        "num_groups": num_groups,
        "weight": weight,
        "bias": bias,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.randn(1, 12, 8, 8).astype(np.float32)
    num_groups = 4
    weight = np.random.randn(12).astype(np.float32)
    bias = np.random.randn(12).astype(np.float32)
    eps = 1e-8

    input_dict = {
        "input": input_tensor,
        "num_groups": num_groups,
        "weight": weight,
        "bias": bias,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.randn(4, 8, 2, 2).astype(np.float32)
    num_groups = 1
    weight = np.random.randn(8).astype(np.float32)
    bias = np.random.randn(8).astype(np.float32)
    eps = 1e-3

    input_dict = {
        "input": input_tensor,
        "num_groups": num_groups,
        "weight": weight,
        "bias": bias,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input_tensor = np.random.randn(3, 9, 5, 5).astype(np.float32)
    num_groups = 3
    weight = np.random.randn(9).astype(np.float32)
    bias = np.random.randn(9).astype(np.float32)
    eps = 1e-6

    input_dict = {
        "input": input_tensor,
        "num_groups": num_groups,
        "weight": weight,
        "bias": bias,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.random.randn(1, 16, 16).astype(np.float32)
    num_groups = 8
    weight = np.random.randn(16).astype(np.float32)
    bias = np.random.randn(16).astype(np.float32)
    eps = 1e-7
    input_dict = {
        "input": input_tensor,
        "num_groups": num_groups,
        "weight": weight,
        "bias": bias,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.random.randn(2, 4, 3).astype(np.float32)
    num_groups = 2
    weight = np.random.randn(4).astype(np.float32)
    bias = np.random.randn(4).astype(np.float32)
    eps = 1e-4
    input_dict = {
        "input": input_tensor,
        "num_groups": num_groups,
        "weight": weight,
        "bias": bias,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.random.randn(1, 32).astype(np.float32)
    num_groups = 4
    weight = np.random.randn(32).astype(np.float32)
    bias = np.random.randn(32).astype(np.float32)
    eps = 1e-2
    input_dict = {
        "input": input_tensor,
        "num_groups": num_groups,
        "weight": weight,
        "bias": bias,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.random.randn(1, 4, 4, 4, 4).astype(np.float32)
    num_groups = 2
    weight = np.random.randn(4).astype(np.float32)
    bias = np.random.randn(4).astype(np.float32)
    eps = 1e-5
    input_dict = {
        "input": input_tensor,
        "num_groups": num_groups,
        "weight": weight,
        "bias": bias,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 - different input size
    input_tensor = np.random.randn(3, 12, 10, 10).astype(np.float32)
    num_groups = 6
    weight = np.random.randn(12).astype(np.float32)
    bias = np.random.randn(12).astype(np.float32)
    eps = 1e-5

    input_dict = {
        "input": input_tensor,
        "num_groups": num_groups,
        "weight": weight,
        "bias": bias,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 - different eps
    input_tensor = np.random.randn(2, 8, 5, 5).astype(np.float32)
    num_groups = 4
    weight = np.random.randn(8).astype(np.float32)
    bias = np.random.randn(8).astype(np.float32)
    eps = 1e-9

    input_dict = {
        "input": input_tensor,
        "num_groups": num_groups,
        "weight": weight,
        "bias": bias,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.group_norm"] = group_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.group_norm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.group_norm'.")

check_valid('torch.nn.functional.group_norm', generated_inputs['torch.nn.functional.group_norm'], lib="torch", suffix=0)
