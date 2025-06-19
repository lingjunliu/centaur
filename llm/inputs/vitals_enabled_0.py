
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def vitals_enabled_inputs():
    list_of_inputs = []

    input1 = {}
    list_of_inputs.append(copy.deepcopy(input1))

    input2 = {}
    list_of_inputs.append(copy.deepcopy(input2))

    input3 = {}
    list_of_inputs.append(copy.deepcopy(input3))
    
    input4 = {}
    list_of_inputs.append(copy.deepcopy(input4))

    input5 = {}
    list_of_inputs.append(copy.deepcopy(input5))
    
    return list_of_inputs

generated_inputs["torch.vitals_enabled"] = vitals_enabled_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.vitals_enabled' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.vitals_enabled'.")

check_valid('torch.vitals_enabled', generated_inputs['torch.vitals_enabled'], lib="torch")
