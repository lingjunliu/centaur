
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_arange_inputs():
    list_of_inputs = []

    # Case 1: Only end is provided (start=0, step=1)
    input_dict = {
        "start": 0.0,
        "end": 5.0,
        "step": 1.0,
        "out": None,
        "dtype": None,
        "layout": None,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Start, end, and step are provided (integers)
    input_dict = {
        "start": 1.0,
        "end": 10.0,
        "step": 2.0,
        "out": None,
        "dtype": None,
        "layout": None,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Start, end, and step are provided (floats, negative values)
    input_dict = {
        "start": -5.0,
        "end": 5.0,
        "step": 0.5,
        "out": None,
        "dtype": None,
        "layout": None,
        "requires_grad": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Using a specific dtype (torch.int32)
    input_dict = {
        "start": 0.0,
        "end": 10.0,
        "step": 1.0,
        "out": None,
        "dtype": torch.int32,
        "layout": None,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Different values
    input_dict = {
        "start": 2.0,
        "end": 7.0,
        "step": 0.5,
        "out": None,
        "dtype": None,
        "layout": None,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.arange"] = torch_arange_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.arange' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.arange'.")

check_valid('torch.arange', generated_inputs['torch.arange'], lib="torch")
