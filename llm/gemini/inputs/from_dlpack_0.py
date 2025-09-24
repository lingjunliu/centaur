
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def from_dlpack_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    x = torch.randn(3, 4).numpy()
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer tensor with negative values
    x = torch.randint(-10, 10, (2, 2), dtype=torch.int32).numpy()
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D tensor
    x = torch.arange(5).numpy()
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 3D tensor
    x = torch.randn(2, 3, 4).numpy()
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Double tensor
    x = torch.randn(2, 3, dtype=torch.float64).numpy()
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Bool tensor
    x = torch.randint(0, 2, (3, 3)).bool().numpy()
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Complex tensor
    x = torch.randn(2, 3, dtype=torch.complex64).numpy()
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.from_dlpack"] = from_dlpack_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.from_dlpack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.from_dlpack'.")

check_valid('torch.from_dlpack', generated_inputs['torch.from_dlpack'], lib="torch")
