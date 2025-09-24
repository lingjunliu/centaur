
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def as_tensor_inputs():
    list_of_inputs = []

    # Input 1: Integer numpy array
    data = np.array([1, 2, 3, 4, 5])
    input_dict = {"data": data, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float numpy array
    data = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    input_dict = {"data": data, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-dimensional integer numpy array
    data = np.array([[1, 2], [3, 4]])
    input_dict = {"data": data, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Numpy array with specified dtype
    data = np.array([1, 2, 3], dtype=np.int64)
    input_dict = {"data": data, "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Numpy array with negative values
    data = np.array([-1, 0, 1])
    input_dict = {"data": data, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Higher dimension numpy array
    data = np.random.rand(2, 3, 4)
    input_dict = {"data": data, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex numpy array
    data = np.array([1+1j, 2+2j, 3+3j])
    input_dict = {"data": data, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.as_tensor_4"] = as_tensor_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.as_tensor_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.as_tensor_4'.")

check_valid('torch.as_tensor', generated_inputs['torch.as_tensor_4'], lib="torch")
