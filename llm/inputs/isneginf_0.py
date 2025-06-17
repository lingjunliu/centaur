
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def isneginf_inputs():
    list_of_inputs = []

    # Case 1: 1D tensor with negative infinity
    input1 = torch.tensor([-float('inf'), 1.0, 2.0])
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: 2D tensor with negative infinity and other values
    input2 = torch.tensor([[float('inf'), -float('inf')], [3.0, -2.0]])
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: 3D tensor with some negative infinity values
    input3 = torch.tensor([[[1.0, -float('inf')], [2.0, 3.0]], [[-float('inf'), 4.0], [5.0, 6.0]]])
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: Tensor with only negative infinity
    input4 = torch.tensor([[-float('inf'), -float('inf')], [-float('inf'), -float('inf')]])
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Case 5: Float 64 tensor
    input5 = torch.tensor([-float('inf'), 1.0, 2.0], dtype=torch.float64)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.isneginf"] = isneginf_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.isneginf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.isneginf'.")

check_valid('torch.isneginf', generated_inputs['torch.isneginf'], lib="torch")
