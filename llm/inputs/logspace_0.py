
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def logspace_inputs():
    list_of_inputs = []

    # Test case 1: Basic test with default base
    input_dict = {
        "start": -10.0,
        "end": 10.0,
        "steps": 5,
        "base": 10.0,
        "out": None,
        "dtype": None,
        "layout": torch.strided,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: Different base
    input_dict = {
        "start": 0.0,
        "end": 5.0,
        "steps": 6,
        "base": 2.0,
        "out": None,
        "dtype": None,
        "layout": torch.strided,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: Small number of steps
    input_dict = {
        "start": 0.1,
        "end": 1.0,
        "steps": 1,
        "base": 10.0,
        "out": None,
        "dtype": None,
        "layout": torch.strided,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: Same start and end
    input_dict = {
        "start": 2.0,
        "end": 2.0,
        "steps": 1,
        "base": 2.0,
        "out": None,
        "dtype": None,
        "layout": torch.strided,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 5: Float start and end, larger number of steps
    input_dict = {
        "start": 0.5,
        "end": 2.5,
        "steps": 10,
        "base": 10.0,
        "out": None,
        "dtype": None,
        "layout": torch.strided,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 6: Negative start and end
    input_dict = {
        "start": -2.0,
        "end": -1.0,
        "steps": 5,
        "base": 10.0,
        "out": None,
        "dtype": None,
        "layout": torch.strided,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.logspace"] = logspace_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.logspace' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logspace'.")

check_valid('torch.logspace', generated_inputs['torch.logspace'], lib="torch")
