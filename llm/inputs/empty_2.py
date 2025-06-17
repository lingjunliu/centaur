
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_empty_inputs():
    list_of_inputs = []

    # Input 1: Basic example with size as a tuple
    input_dict = {
        "size": (2, 3),
        "out": None,
        "dtype": None,
        "layout": torch.strided,
        "requires_grad": False,
        "pin_memory": False,
        "memory_format": torch.contiguous_format,
        "device": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Size as a list, specific dtype
    input_dict = {
        "size": [4, 5, 2],
        "out": None,
        "dtype": torch.float64,
        "layout": torch.strided,
        "requires_grad": True,
        "pin_memory": False,
        "memory_format": torch.contiguous_format,
        "device": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3:  Specific device. Removed pin_memory=True as it might conflict with device.
    input_dict = {
        "size": (1, 8),
        "out": None,
        "dtype": torch.int32,
        "layout": torch.strided,
        "requires_grad": False,
        "pin_memory": False,
        "memory_format": torch.contiguous_format,
        "device": torch.device("cpu")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Using memory_format (only works when out is None)
    input_dict = {
        "size": (2, 2),
        "out": None,
        "dtype": torch.float32,
        "layout": torch.strided,
        "requires_grad": False,
        "pin_memory": False,
        "memory_format": torch.contiguous_format,
        "device": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: No optional parameters
    input_dict = {
        "size": (3, 4),
        "out": None,
        "dtype": None,
        "layout": None,
        "requires_grad": False,
        "pin_memory": False,
        "memory_format": None,
        "device": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.empty_2"] = torch_empty_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.empty_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.empty_2'.")

check_valid('torch.empty', generated_inputs['torch.empty_2'], lib="torch")
