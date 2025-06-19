
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def autocast_decrement_nesting_inputs():
    list_of_inputs = []
    
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.autocast_decrement_nesting"] = autocast_decrement_nesting_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.autocast_decrement_nesting' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.autocast_decrement_nesting'.")

check_valid('torch.autocast_decrement_nesting', generated_inputs['torch.autocast_decrement_nesting'], lib="torch")
