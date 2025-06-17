
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def torch_empty_inputs():
    list_of_inputs = []

    # Input 1: Basic integer size
    input_dict = {
        "size": (2, 3),
        "out": None,
        "dtype": None,
        "layout": None,
        "requires_grad": False,
        "pin_memory": False,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float dtype
    input_dict = {
        "size": (4, 5),
        "out": None,
        "dtype": torch.float64,
        "layout": None,
        "requires_grad": True,
        "pin_memory": False,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Integer dtype
    input_dict = {
        "size": (1, 2, 3),
        "out": None,
        "dtype": torch.int32,
        "layout": None,
        "requires_grad": False,
        "pin_memory": True,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Complex dtype
    input_dict = {
        "size": (2, 2),
        "out": None,
        "dtype": torch.complex64,
        "layout": None,
        "requires_grad": False,
        "pin_memory": False,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different size format (list instead of tuple)
    input_dict = {
        "size": [5, 5, 5],
        "out": None,
        "dtype": None,
        "layout": None,
        "requires_grad": False,
        "pin_memory": False,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Using out parameter
    out_tensor = torch.randn(3, 4)
    input_dict = {
        "size": (3, 4),
        "out": out_tensor.numpy(),
        "dtype": None,
        "layout": None,
        "requires_grad": False,
        "pin_memory": False,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Bool dtype
    input_dict = {
        "size": (2, 2),
        "out": None,
        "dtype": torch.bool,
        "layout": None,
        "requires_grad": False,
        "pin_memory": False,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.empty_1"] = torch_empty_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.empty_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.empty_1'.")

check_valid('torch.empty', generated_inputs['torch.empty_1'], lib="torch")
