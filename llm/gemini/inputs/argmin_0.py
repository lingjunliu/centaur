
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def argmin_inputs():
    list_of_inputs = []

    # Case 1: 1D tensor, no dim, keepdim=False
    input_tensor = torch.randn(5).numpy()
    input_dict = {"input": input_tensor, "dim": None, "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 2D tensor, dim=0, keepdim=False
    input_tensor = torch.randn(3, 4).numpy()
    input_dict = {"input": input_tensor, "dim": 0, "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 2D tensor, dim=1, keepdim=True
    input_tensor = torch.randn(2, 5).numpy()
    input_dict = {"input": input_tensor, "dim": 1, "keepdim": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 3D tensor, dim=2, keepdim=False
    input_tensor = torch.randn(2, 3, 4).numpy()
    input_dict = {"input": input_tensor, "dim": 2, "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: 3D tensor with negative values, dim=0, keepdim=True
    input_tensor = torch.randn(2, 3, 4).numpy() * -1
    input_dict = {"input": input_tensor, "dim": 0, "keepdim": True}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 6: 4D tensor, dim=1, keepdim=False
    input_tensor = torch.randn(1, 3, 5, 5).numpy()
    input_dict = {"input": input_tensor, "dim": 1, "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Int tensor
    input_tensor = torch.randint(0, 10, (2, 3)).numpy()
    input_dict = {"input": input_tensor, "dim": 0, "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.argmin"] = argmin_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.argmin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.argmin'.")

check_valid('torch.argmin', generated_inputs['torch.argmin'], lib="torch")
