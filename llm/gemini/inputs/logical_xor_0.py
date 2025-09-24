
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def logical_xor_inputs():
    list_of_inputs = []

    # Test case 1: Basic boolean tensors
    input1 = np.array([True, False, True, False])
    input2 = np.array([True, True, False, False])
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: Integer tensors (0 and 1)
    input1 = np.array([1, 0, 1, 0])
    input2 = np.array([1, 1, 0, 0])
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: Multi-dimensional boolean tensors
    input1 = np.array([[True, False], [True, True]])
    input2 = np.array([[False, True], [True, False]])
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 4: Different shapes that are broadcastable
    input1 = np.array([[True, False]])
    input2 = np.array([True, True])
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 5: 3D boolean tensors
    input1 = np.array([[[True, False], [True, True]], [[False, True], [True, False]]])
    input2 = np.array([[[False, True], [True, False]], [[True, False], [False, True]]])
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 6: Using 'out' parameter
    input1 = np.array([True, False, True, False])
    input2 = np.array([True, True, False, False])
    out_array = np.empty_like(input1, dtype=bool)
    input_dict = {"input": input1, "other": input2, "out": out_array}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.logical_xor"] = logical_xor_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.logical_xor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logical_xor'.")

check_valid('torch.logical_xor', generated_inputs['torch.logical_xor'], lib="torch")
