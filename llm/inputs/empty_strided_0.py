
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def empty_strided_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input_dict = {
        "size": (2, 3),
        "stride": (3, 1),
        "dtype": torch.float32
    }
    list_of_inputs.append(input_dict)

    # Input 2: Integer tensor
    input_dict = {
        "size": (4, 5),
        "stride": (5, 1),
        "dtype": torch.int64
    }
    list_of_inputs.append(input_dict)

    # Input 3: Complex tensor
    input_dict = {
        "size": (2, 2, 2),
        "stride": (4, 2, 1),
        "dtype": torch.complex64
    }
    list_of_inputs.append(input_dict)

    # Input 4: Boolean tensor
    input_dict = {
        "size": (1, 5, 5),
        "stride": (25, 5, 1),
        "dtype": torch.bool
    }
    list_of_inputs.append(input_dict)

    # Input 5: High dimension tensor
    input_dict = {
        "size": (2, 2, 2, 2),
        "stride": (8, 4, 2, 1),
        "dtype": torch.float64
    }
    list_of_inputs.append(input_dict)
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.empty_strided"] = empty_strided_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.empty_strided' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.empty_strided'.")

check_valid('torch.empty_strided', generated_inputs['torch.empty_strided'], lib="torch")
