
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_ones_inputs():
    list_of_inputs = []

    # Input 1: Basic example with different size
    input_dict = {
        "size": (2, 3),
        "out": None,
        "dtype": None,
        "layout": torch.strided,
        "requires_grad": False
    }
    list_of_inputs.append(input_dict)

    # Input 2: 1D tensor
    input_dict = {
        "size": [5],
        "out": None,
        "dtype": None,
        "layout": torch.strided,
        "requires_grad": False
    }
    list_of_inputs.append(input_dict)

    # Input 3: Specifying dtype
    input_dict = {
        "size": (2, 2),
        "out": None,
        "dtype": torch.int32,
        "layout": torch.strided,
        "requires_grad": False
    }
    list_of_inputs.append(input_dict)

    # Input 4: requires_grad = True
    input_dict = {
        "size": (3, 4),
        "out": None,
        "dtype": None,
        "layout": torch.strided,
        "requires_grad": True
    }
    list_of_inputs.append(input_dict)

    # Input 5: Using a list for size
    input_dict = {
        "size": [2, 5],
        "out": None,
        "dtype": None,
        "layout": torch.strided,
        "requires_grad": False
    }
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs["torch.ones_3"] = torch_ones_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.ones_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.ones_3'.")

check_valid('torch.ones', generated_inputs['torch.ones_3'], lib="torch")
