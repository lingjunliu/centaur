
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def neg_inputs():
    list_of_inputs = []

    # Example 1: 1D float tensor with positive and negative values
    input1 = np.array([1.0, -2.0, 3.0, -4.0, 5.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Example 2: 2D int tensor with mixed values
    input2 = np.array([[1, -2], [3, -4]], dtype=np.int32)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Example 3: 3D float tensor with mixed values
    input3 = np.array([[[1.0, -2.0], [3.0, -4.0]], [[5.0, -6.0], [7.0, -8.0]]], dtype=np.float64)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Example 4: 1D complex tensor
    input4 = np.array([1+1j, 2-2j, 3+0j], dtype=np.complex64)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Example 5: scalar tensor
    input5 = np.array(-5, dtype=np.int64)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.neg"] = neg_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.neg' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.neg'.")

check_valid('torch.neg', generated_inputs['torch.neg'], lib="torch")
