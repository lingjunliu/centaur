
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def torch_zeros_inputs():
    list_of_inputs = []

    # Test case 1: Basic test with a tuple size
    input_dict = {
        "size": (2, 3),
        "out": None,
        "dtype": None,
        "layout": torch.strided,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: Basic test with a list size
    input_dict = {
        "size": [5],
        "out": None,
        "dtype": None,
        "layout": torch.strided,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: Test with specific dtype
    input_dict = {
        "size": (4, 2),
        "out": None,
        "dtype": torch.int32,
        "layout": torch.strided,
        "requires_grad": False # Integer tensors cannot require gradients
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: Test with an out tensor
    out_tensor = torch.empty(3, 4)
    input_dict = {
        "size": (3, 4),
        "out": out_tensor,
        "dtype": None,
        "layout": torch.strided,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 5: Test with different dtype and requires_grad
    input_dict = {
        "size": (2, 2, 2),
        "out": None,
        "dtype": torch.float64,
        "layout": torch.strided,
        "requires_grad": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.zeros_1"] = torch_zeros_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.zeros_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.zeros_1'.")

check_valid('torch.zeros', generated_inputs['torch.zeros_1'], lib="torch")
