
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def igammac_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensors
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    other1 = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: Integer tensors (converted to float)
    input2 = np.array([1, 2, 3], dtype=np.int32).astype(np.float32)
    other2 = np.array([1, 1, 1], dtype=np.int32).astype(np.float32)
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: Multi-dimensional tensors
    input3 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    other3 = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64)
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: Broadcastable shapes
    input4 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    other4 = np.array([0.5, 1.5], dtype=np.float32)
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Case 5: Different data types
    input5 = np.array([1.0, 2.0], dtype=np.float64)
    other5 = np.array([0.5, 1.5], dtype=np.float32)
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.igammac"] = igammac_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.igammac' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.igammac'.")

check_valid('torch.igammac', generated_inputs['torch.igammac'], lib="torch")
