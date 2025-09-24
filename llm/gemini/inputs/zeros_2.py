
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def torch_zeros_inputs():
    list_of_inputs = []

    # Input 1: Basic size
    input_dict = {
        "size": (2, 3),
        "out": None,
        "dtype": None,
        "layout": torch.strided,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different size and dtype
    input_dict = {
        "size": (5,),
        "out": None,
        "dtype": torch.int32,
        "layout": torch.strided,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Using a list for size, requires_grad=True
    input_dict = {
        "size": (4, 2, 1),
        "out": None,
        "dtype": None,
        "layout": torch.strided,
        "requires_grad": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different dtype and layout
    input_dict = {
        "size": (3, 4),
        "out": None,
        "dtype": torch.complex64,
        "layout": torch.strided,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.zeros_2"] = torch_zeros_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.zeros_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.zeros_2'.")

check_valid('torch.zeros', generated_inputs['torch.zeros_2'], lib="torch")
