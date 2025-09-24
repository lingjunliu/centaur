
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def outer_inputs():
    list_of_inputs = []

    # Example 1: Basic float tensors
    v1 = np.arange(1., 5.)
    v2 = np.arange(1., 4.)
    input_dict = {"input": v1, "vec2": v2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Integer tensors
    v1 = np.arange(1, 6)
    v2 = np.arange(1, 5)
    input_dict = {"input": v1, "vec2": v2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: Negative values
    v1 = np.array([-1, 2, -3, 4])
    v2 = np.array([5, -6, 7])
    input_dict = {"input": v1, "vec2": v2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: Different data types (float64)
    v1 = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    v2 = np.array([4.5, 5.5], dtype=np.float64)
    input_dict = {"input": v1, "vec2": v2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5: Zero values
    v1 = np.array([0, 1, 2, 3])
    v2 = np.array([0, 4, 5])
    input_dict = {"input": v1, "vec2": v2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 6: All same values
    v1 = np.array([2, 2, 2])
    v2 = np.array([3, 3])
    input_dict = {"input": v1, "vec2": v2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 7: One element tensors
    v1 = np.array([5])
    v2 = np.array([6])
    input_dict = {"input": v1, "vec2": v2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.outer"] = outer_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.outer' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.outer'.")

check_valid('torch.outer', generated_inputs['torch.outer'], lib="torch")
