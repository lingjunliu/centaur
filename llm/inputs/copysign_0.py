
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def copysign_inputs():
    list_of_inputs = []

    # Test case 1: Basic test with positive and negative values
    input1 = torch.randn(5).numpy()
    other1 = torch.randn(5).numpy()
    input_dict1 = {"input": input1, "other": other1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Test case 2:  'other' is a scalar
    input2 = torch.randn(3, 3).numpy()
    other2 = -1.0
    input_dict2 = {"input": input2, "other": other2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Test case 3: 'input' is all zeros
    input3 = torch.zeros(2, 2).numpy()
    other3 = torch.randn(2, 2).numpy()
    input_dict3 = {"input": input3, "other": other3, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Test case 4: 'other' is all zeros, demonstrating signed zero handling
    input4 = torch.randn(4).numpy()
    other4 = torch.zeros(4).numpy()
    input_dict4 = {"input": input4, "other": other4, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Test case 5: Broadcasting 'other'
    input5 = torch.randn(2, 3, 4).numpy()
    other5 = torch.randn(4).numpy()
    input_dict5 = {"input": input5, "other": other5, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.copysign"] = copysign_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.copysign' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.copysign'.")

check_valid('torch.copysign', generated_inputs['torch.copysign'], lib="torch")
