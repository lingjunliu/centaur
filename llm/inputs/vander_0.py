
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def vander_inputs():
    list_of_inputs = []

    # Case 1: Basic case with default N and increasing
    x = np.array([1, 2, 3, 4])
    input_dict = {"x": x, "N": None, "increasing": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Specify N
    x = np.array([1, 2, 3, 4])
    input_dict = {"x": x, "N": 3, "increasing": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Increasing is True
    x = np.array([1, 2, 3, 4])
    input_dict = {"x": x, "N": 3, "increasing": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Different data type (float)
    x = np.array([1.5, 2.5, 3.5])
    input_dict = {"x": x, "N": None, "increasing": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Different data type (int64) and negative values
    x = np.array([-1, 2, -3, 4], dtype=np.int64)
    input_dict = {"x": x, "N": 5, "increasing": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 6: Single element tensor
    x = np.array([5])
    input_dict = {"x": x, "N": None, "increasing": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Empty tensor
    x = np.array([])
    input_dict = {"x": x, "N": 3, "increasing": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.vander"] = vander_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.vander' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.vander'.")

check_valid('torch.vander', generated_inputs['torch.vander'], lib="torch")
