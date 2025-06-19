
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def autocast_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "device_type": "cuda",
        "dtype": "float16",
        "enabled": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "device_type": "cpu",
        "dtype": "float32",
        "enabled": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "device_type": "cuda",
        "dtype": "bfloat16",
        "enabled": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "device_type": "cpu",
        "dtype": "float64",
        "enabled": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "device_type": "cuda",
        "dtype": "float64",
        "enabled": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.autocast_2"] = autocast_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.autocast_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.autocast_2'.")

check_valid('torch.autocast', generated_inputs['torch.autocast_2'], lib="torch", suffix=2)
