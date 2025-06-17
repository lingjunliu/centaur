
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_atan_inputs():
    list_of_inputs = []

    # Input 1: 1D float tensor
    input1 = torch.randn(4).numpy()
    input_dict1 = {"input": input1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D float tensor with negative values
    input2 = torch.randn(2, 3).numpy() * -1
    input_dict2 = {"input": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D float tensor
    input3 = torch.randn(2, 2, 2).numpy()
    input_dict3 = {"input": input3, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 1D int tensor
    input4 = torch.randint(-5, 5, (5,)).numpy()
    input_dict4 = {"input": input4, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 2D int tensor with zeros
    input5 = torch.randint(-2, 2, (3, 3)).numpy()
    input_dict5 = {"input": input5, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.atan"] = torch_atan_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.atan' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.atan'.")

check_valid('torch.atan', generated_inputs['torch.atan'], lib="torch")
