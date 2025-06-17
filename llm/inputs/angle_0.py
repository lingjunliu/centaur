
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def angle_inputs():
    list_of_inputs = []

    # Example 1: 1D complex tensor
    input1 = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j], dtype=torch.complex64)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Example 2: 2D complex tensor
    input2 = torch.tensor([[1 + 1j, 2 - 2j], [-3 - 3j, 4 + 0j]], dtype=torch.complex64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Example 3: 1D real tensor (positive and negative)
    input3 = torch.tensor([-1.0, -2.0, 3.0, 4.0, 0.0], dtype=torch.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Example 4: 2D real tensor (positive and negative)
    input4 = torch.tensor([[-1.0, 2.0], [-3.0, 4.0]], dtype=torch.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Example 5: Scalar complex value
    input5 = torch.tensor(1 + 1j, dtype=torch.complex64)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.angle"] = angle_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.angle' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.angle'.")

check_valid('torch.angle', generated_inputs['torch.angle'], lib="torch")
