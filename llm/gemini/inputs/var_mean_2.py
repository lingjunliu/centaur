
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def var_mean_inputs():
    list_of_inputs = []

    # Input 6: Removing the 'out' parameter and adjusting types for valid input
    input_tensor = torch.randn(5).numpy()
    dim = [0]
    unbiased = False
    keepdim = False
    input_dict = {
        "input": input_tensor,
        "dim": [0],
        "unbiased": unbiased,
        "keepdim": keepdim,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Another valid input without 'out'
    input_tensor = torch.randn(2, 3).numpy()
    dim = [0, 1]
    unbiased = True
    keepdim = True
    input_dict = {
        "input": input_tensor,
        "dim": [0, 1],
        "unbiased": unbiased,
        "keepdim": keepdim,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Another valid input with single dimension
    input_tensor = torch.randn(4, 4).numpy()
    dim = [1]
    unbiased = False
    keepdim = False
    input_dict = {
        "input": input_tensor,
        "dim": [1],
        "unbiased": unbiased,
        "keepdim": keepdim,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = torch.randn(3, 4).numpy()
    dim = []
    unbiased = True
    keepdim = False
    input_dict = {
        "input": input_tensor,
        "dim": [],
        "unbiased": unbiased,
        "keepdim": keepdim,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = torch.randn(1, 2, 3).numpy()
    dim = [0, 2]
    unbiased = False
    keepdim = True
    input_dict = {
        "input": input_tensor,
        "dim": [0, 2],
        "unbiased": unbiased,
        "keepdim": keepdim,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Testing negative values and different dimensions
    input_tensor = torch.randn(2, 2) * -1.0
    input_tensor = input_tensor.numpy()
    dim = [0]
    unbiased = True
    keepdim = False
    input_dict = {
        "input": input_tensor,
        "dim": [0],
        "unbiased": unbiased,
        "keepdim": keepdim,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Empty dim list
    input_tensor = torch.randn(3, 3).numpy()
    dim = []
    unbiased = False
    keepdim = True
    input_dict = {
        "input": input_tensor,
        "dim": [],
        "unbiased": unbiased,
        "keepdim": keepdim,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13: Ensure dim is a list of numpy int64
    input_tensor = torch.randn(4, 5).numpy()
    dim = np.array([0, 1], dtype=np.int64).tolist()
    unbiased = True
    keepdim = False
    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "unbiased": unbiased,
        "keepdim": keepdim,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.var_mean_2"] = var_mean_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.var_mean_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.var_mean_2'.")

check_valid('torch.var_mean', generated_inputs['torch.var_mean_2'], lib="torch", suffix=2)
