
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_empty_inputs():
    list_of_inputs = []

    # Input 1: Basic case with integer size
    input_dict = {
        "size": [2, 3],
        "out": None,
        "dtype": None,
        "layout": torch.strided,
        "requires_grad": False,
        "pin_memory": False,
        "memory_format": torch.contiguous_format
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float dtype
    input_dict = {
        "size": [4, 5],
        "out": None,
        "dtype": torch.float64,
        "layout": torch.strided,
        "requires_grad": True,
        "pin_memory": False,
        "memory_format": torch.contiguous_format
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different layout
    input_dict = {
        "size": [1, 7, 2],
        "out": None,
        "dtype": torch.int32,
        "layout": torch.strided,
        "requires_grad": False,
        "pin_memory": True,
        "memory_format": torch.contiguous_format
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Complex dtype
    input_dict = {
        "size": [3, 3, 3],
        "out": None,
        "dtype": torch.complex64,
        "layout": torch.strided,
        "requires_grad": False,
        "pin_memory": False,
        "memory_format": torch.contiguous_format
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty list as size
    input_dict = {
        "size": [],
        "out": None,
        "dtype": torch.int8,
        "layout": torch.strided,
        "requires_grad": False,
        "pin_memory": False,
        "memory_format": torch.contiguous_format
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.empty_3"] = torch_empty_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.empty_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.empty_3'.")

check_valid('torch.empty', generated_inputs['torch.empty_3'], lib="torch")
