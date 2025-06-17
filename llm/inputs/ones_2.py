
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_ones_inputs():
    list_of_inputs = []

    # Input 1: Basic example with different size
    input_dict = {
        "size": (2, 3),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different size with a list
    input_dict = {
        "size": [5],
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Using dtype
    input_dict = {
        "size": (4, 2),
        "dtype": torch.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: requires_grad=True
    input_dict = {
        "size": (3, 3),
        "requires_grad": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.ones_2"] = torch_ones_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.ones_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.ones_2'.")

check_valid('torch.ones', generated_inputs['torch.ones_2'], lib="torch")
