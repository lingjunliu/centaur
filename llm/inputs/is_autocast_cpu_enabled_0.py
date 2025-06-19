
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy

def is_autocast_cpu_enabled_inputs():
    list_of_inputs = []

    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.is_autocast_cpu_enabled"] = is_autocast_cpu_enabled_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.is_autocast_cpu_enabled' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.is_autocast_cpu_enabled'.")

check_valid('torch.is_autocast_cpu_enabled', generated_inputs['torch.is_autocast_cpu_enabled'], lib="torch")
