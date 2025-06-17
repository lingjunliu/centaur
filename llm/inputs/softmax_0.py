
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def softmax_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D float tensor
    input1 = np.random.randn(3, 4).astype(np.float32)
    dim1 = 1
    dtype1 = None
    input_dict1 = {"input": input1, "dim": dim1, "dtype": dtype1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 3D float tensor with negative values
    input2 = np.random.randn(2, 3, 5).astype(np.float64) * -1
    dim2 = 2
    dtype2 = None
    input_dict2 = {"input": input2, "dim": dim2, "dtype": dtype2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 1D float tensor
    input3 = np.random.randn(6).astype(np.float32)
    dim3 = 0
    dtype3 = None
    input_dict3 = {"input": input3, "dim": dim3, "dtype": dtype3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 2D float tensor with specified dtype
    input4 = np.random.randn(4, 4).astype(np.float64)
    dim4 = 0
    dtype4 = torch.float32
    input_dict4 = {"input": input4, "dim": dim4, "dtype": dtype4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 4D float tensor
    input5 = np.random.randn(2, 2, 2, 2).astype(np.float32)
    dim5 = 3
    dtype5 = None
    input_dict5 = {"input": input5, "dim": dim5, "dtype": dtype5}
    list_of_inputs.append(copy.deepcopy(input_dict5))


    return list_of_inputs

generated_inputs["torch.special.softmax"] = softmax_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.softmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.softmax'.")

check_valid('torch.special.softmax', generated_inputs['torch.special.softmax'], lib="torch")
