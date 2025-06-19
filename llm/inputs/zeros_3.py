
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def torch_zeros_inputs():
    list_of_inputs = []

    # Case 1: Basic size (tuple)
    input_dict = {
        "size": (2, 3),
        "out": None,
        "dtype": None,
        "layout": torch.strided,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Basic size (list)
    input_dict = {
        "size": [4, 5],
        "out": None,
        "dtype": None,
        "layout": torch.strided,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Single dimension
    input_dict = {
        "size": (7,),
        "out": None,
        "dtype": None,
        "layout": torch.strided,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 4: Explicit dtype (torch.float64)
    input_dict = {
        "size": (2, 2),
        "out": None,
        "dtype": torch.float64,
        "layout": torch.strided,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Explicit dtype (torch.float64) and requires_grad=True
    input_dict = {
        "size": (3, 4),
        "out": None,
        "dtype": torch.float64,
        "layout": torch.strided,
        "requires_grad": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.zeros_3"] = torch_zeros_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.zeros_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.zeros_3'.")

check_valid('torch.zeros', generated_inputs['torch.zeros_3'], lib="torch")
