
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def log2_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input1 = np.random.rand(5).astype(np.float32)
    input_dict1 = {"input": input1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Tensor with negative values (should return NaN for negative values)
    input2 = np.array([-1.0, 0.5, 2.0, 4.0, 8.0]).astype(np.float64)
    input_dict2 = {"input": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Multi-dimensional tensor (2D)
    input3 = np.random.rand(2, 3).astype(np.float32)
    input_dict3 = {"input": input3, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Tensor with zeros (should return -inf)
    input4 = np.array([0.0, 1.0, 2.0]).astype(np.float32)
    input_dict4 = {"input": input4, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Larger tensor
    input5 = np.random.rand(4, 4, 4).astype(np.float64)
    input_dict5 = {"input": input5, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Scalar tensor
    input6 = np.array(2.0).astype(np.float32)
    input_dict6 = {"input": input6, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Tensor with very small values close to zero
    input7 = np.array([1e-8, 1e-7, 1e-6]).astype(np.float32)
    input_dict7 = {"input": input7, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.log2"] = log2_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.log2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.log2'.")

check_valid('torch.log2', generated_inputs['torch.log2'], lib="torch")
