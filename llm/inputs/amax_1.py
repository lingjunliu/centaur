
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def amax_inputs():
    list_of_inputs = []

    # Input 1: Basic case with dim=1 and keepdim=False
    input1 = torch.randn(4, 4).numpy()
    dim1 = 1
    keepdim1 = False
    input_dict1 = {"input": input1, "dim": dim1, "keepdim": keepdim1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Different dim and keepdim=True
    input2 = torch.randn(2, 3, 5).numpy()
    dim2 = 0
    keepdim2 = True
    input_dict2 = {"input": input2, "dim": dim2, "keepdim": keepdim2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Multiple dimensions
    input3 = torch.randn(2, 3, 4, 5).numpy()
    dim3 = (1, 2)
    keepdim3 = False
    input_dict3 = {"input": input3, "dim": dim3, "keepdim": keepdim3, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Negative dimension
    input4 = torch.randn(3, 5, 2).numpy()
    dim4 = -1
    keepdim4 = True
    input_dict4 = {"input": input4, "dim": dim4, "keepdim": keepdim4, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Integer tensor
    input5 = torch.randint(0, 10, (2, 2, 2)).numpy()
    dim5 = 0
    keepdim5 = False
    input_dict5 = {"input": input5, "dim": dim5, "keepdim": keepdim5, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.amax_1"] = amax_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.amax_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.amax_1'.")

check_valid('torch.amax', generated_inputs['torch.amax_1'], lib="torch")
