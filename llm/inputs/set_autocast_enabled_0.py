
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def set_autocast_enabled_inputs():
    list_of_inputs = []
    
    input_dict = {
        "enabled": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "enabled": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "enabled": bool(np.bool_(True))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "enabled": bool(np.bool_(False))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "enabled": bool(np.array(True))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.set_autocast_enabled"] = set_autocast_enabled_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.set_autocast_enabled' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.set_autocast_enabled'.")

check_valid('torch.set_autocast_enabled', generated_inputs['torch.set_autocast_enabled'], lib="torch")
