
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def as_tensor_inputs():
    list_of_inputs = []

    # Input 1: Simple float array
    data = np.array([1.0, 2.0, 3.0])
    input_dict = {"data": data, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer array with specified dtype
    data = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {"data": data, "dtype": torch.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-dimensional array
    data = np.array([[1, 2], [3, 4]])
    input_dict = {"data": data, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Array with negative values
    data = np.array([-1.0, 0.0, 1.0])
    input_dict = {"data": data, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Array with a different dtype
    data = np.array([1, 2, 3], dtype=np.int64)
    input_dict = {"data": data, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Boolean array
    data = np.array([True, False, True])
    input_dict = {"data": data, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Zero dimensional numpy array
    data = np.array(5)
    input_dict = {"data": data, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.as_tensor_2"] = as_tensor_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.as_tensor_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.as_tensor_2'.")

check_valid('torch.as_tensor', generated_inputs['torch.as_tensor_2'], lib="torch")
