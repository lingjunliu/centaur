
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def argmax_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor
    input1 = torch.randn(5).numpy()
    input_dict1 = {"input": input1, "dim": None, "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor, dim=0
    input2 = torch.randn(3, 4).numpy()
    input_dict2 = {"input": input2, "dim": 0, "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 2D tensor, dim=1, keepdim=True
    input3 = torch.randn(2, 5).numpy()
    input_dict3 = {"input": input3, "dim": 1, "keepdim": True}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 3D tensor, dim=2
    input4 = torch.randn(2, 3, 4).numpy()
    input_dict4 = {"input": input4, "dim": 2, "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: 4D tensor, dim=3, keepdim=True
    input5 = torch.randn(1, 2, 3, 4).numpy()
    input_dict5 = {"input": input5, "dim": 3, "keepdim": True}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Tensor with negative values
    input6 = torch.randint(-10, 10, (2, 3)).float().numpy()
    input_dict6 = {"input": input6, "dim": 1, "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Int Tensor
    input7 = torch.randint(0, 10, (3, 2)).int().numpy()
    input_dict7 = {"input": input7, "dim": 0, "keepdim": True}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs["torch.argmax_2"] = argmax_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.argmax_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.argmax_2'.")

check_valid('torch.argmax', generated_inputs['torch.argmax_2'], lib="torch")
