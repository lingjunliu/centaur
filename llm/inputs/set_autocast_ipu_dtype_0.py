
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def set_autocast_ipu_dtype_inputs():
    list_of_inputs = []

    input_dict = {
        "dtype": torch.float16
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "dtype": torch.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "dtype": torch.int8
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "dtype": torch.int16
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "dtype": torch.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.set_autocast_ipu_dtype"] = set_autocast_ipu_dtype_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.set_autocast_ipu_dtype' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.set_autocast_ipu_dtype'.")

check_valid('torch.set_autocast_ipu_dtype', generated_inputs['torch.set_autocast_ipu_dtype'], lib="torch")
