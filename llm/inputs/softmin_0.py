
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def softmin_inputs():
    list_of_inputs = []

    # Test case 1: Basic 2D tensor with dim=1
    input1 = torch.randn(2, 3).numpy()
    dim1 = 1
    dtype1 = None
    input_dict1 = {"input": input1, "dim": dim1, "dtype": dtype1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Test case 2: 3D tensor with dim=0
    input2 = torch.randn(3, 4, 5).numpy()
    dim2 = 0
    dtype2 = None
    input_dict2 = {"input": input2, "dim": dim2, "dtype": dtype2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Test case 3: 1D tensor with dim=0
    input3 = torch.randn(5).numpy()
    dim3 = 0
    dtype3 = None
    input_dict3 = {"input": input3, "dim": dim3, "dtype": dtype3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Test case 4: 2D tensor with dim=-1 (last dimension)
    input4 = torch.randn(4, 6).numpy()
    dim4 = -1
    dtype4 = None
    input_dict4 = {"input": input4, "dim": dim4, "dtype": dtype4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Test case 5: 2D tensor with dtype specified (torch.float64)
    input5 = torch.randn(2, 2).numpy()
    dim5 = 1
    dtype5 = torch.float64
    input_dict5 = {"input": input5, "dim": dim5, "dtype": dtype5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Test case 6: Tensor with negative values
    input6 = torch.randn(3, 3) * -1
    input6 = input6.numpy()
    dim6 = 1
    dtype6 = None
    input_dict6 = {"input": input6, "dim": dim6, "dtype": dtype6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Test case 7: 4D Tensor
    input7 = torch.randn(2, 3, 4, 5).numpy()
    dim7 = 2
    dtype7 = None
    input_dict7 = {"input": input7, "dim": dim7, "dtype": dtype7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.softmin"] = softmin_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.softmin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.softmin'.")

check_valid('torch.nn.functional.softmin', generated_inputs['torch.nn.functional.softmin'], lib="torch")
