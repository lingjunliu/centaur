
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def set_deterministic_debug_mode_inputs():
    list_of_inputs = []

    input_dict = {
        "mode": "default"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "mode": "warn"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "mode": "error"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.set_deterministic_debug_mode"] = set_deterministic_debug_mode_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.set_deterministic_debug_mode' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.set_deterministic_debug_mode'.")

check_valid('torch.set_deterministic_debug_mode', generated_inputs['torch.set_deterministic_debug_mode'], lib="torch")
