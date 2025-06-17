
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np
from torch.nn import Parameter, ParameterList

def ParameterList_inputs():
    list_of_inputs = []

    # Example 1: List of float tensors
    values = [Parameter(torch.randn(2, 3)) for _ in range(3)]
    input_dict = {"values": values}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: List of int tensors
    values = [Parameter(torch.randint(0, 10, (4, 4)).float()) for _ in range(2)]
    input_dict = {"values": values}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: List of mixed float and int tensors
    values = [Parameter(torch.randn(5)), Parameter(torch.randint(0, 5, (2,)).float())]
    input_dict = {"values": values}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: List of tensors with different dimensions
    values = [Parameter(torch.randn(1, 2, 3)), Parameter(torch.randn(4, 5))]
    input_dict = {"values": values}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5: Empty list
    values = []
    input_dict = {"values": values}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.ParameterList"] = ParameterList_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.ParameterList' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ParameterList'.")

check_valid('torch.nn.ParameterList', generated_inputs['torch.nn.ParameterList'], lib="torch")
