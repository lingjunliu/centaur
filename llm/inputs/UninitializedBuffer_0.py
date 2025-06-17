
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def uninitializedbuffer_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input_dict = {
        "size": (2, 3, 4),
        "dtype": torch.float32,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer tensor
    input_dict = {
        "size": (5, 5),
        "dtype": torch.int64,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Complex tensor
    input_dict = {
        "size": (2, 2, 2),
        "dtype": torch.complex64,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Boolean tensor
    input_dict = {
        "size": (10,),
        "dtype": torch.bool,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.UninitializedBuffer"] = uninitializedbuffer_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.UninitializedBuffer' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.UninitializedBuffer'.")

check_valid('torch.nn.UninitializedBuffer', generated_inputs['torch.nn.UninitializedBuffer'], lib="torch")
