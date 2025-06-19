
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def full_inputs():
    list_of_inputs = []

    # Test case 1: Basic case with integer size and float fill_value
    size = (2, 3)
    fill_value = 3.14
    input_dict = {
        "size": size,
        "fill_value": fill_value,
        "out": None,
        "dtype": None,
        "layout": torch.strided,
        "device": None,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: Different size, integer fill_value, and requires_grad=True
    size = (4, 5, 2)
    fill_value = 5.0
    input_dict = {
        "size": size,
        "fill_value": fill_value,
        "out": None,
        "dtype": torch.float32,
        "layout": torch.strided,
        "device": None,
        "requires_grad": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: Size as a torch.Size object, and specify dtype
    size = torch.Size([1, 6])
    fill_value = 2.718
    input_dict = {
        "size": tuple(size),
        "fill_value": fill_value,
        "out": None,
        "dtype": torch.float64,
        "layout": torch.strided,
        "device": None,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: Single dimension size, negative fill_value
    size = (7,)
    fill_value = -1.0
    input_dict = {
        "size": size,
        "fill_value": fill_value,
        "out": None,
        "dtype": None,
        "layout": torch.strided,
        "device": None,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 5: Empty size, zero fill_value
    size = (0,)
    fill_value = 0.0
    input_dict = {
        "size": size,
        "fill_value": fill_value,
        "out": None,
        "dtype": None,
        "layout": torch.strided,
        "device": None,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.full"] = full_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.full' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.full'.")

check_valid('torch.full', generated_inputs['torch.full'], lib="torch")
