
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy

def set_fusion_strategy_inputs():
    list_of_inputs = []

    input_dict = {
        "value": [("STATIC", 0)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "value": [("DYNAMIC", 0)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "value": [("STATIC", 1)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "value": [("DYNAMIC", 1)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.jit.set_fusion_strategy"] = set_fusion_strategy_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.jit.set_fusion_strategy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.jit.set_fusion_strategy'.")

check_valid('torch.jit.set_fusion_strategy', generated_inputs['torch.jit.set_fusion_strategy'], lib="torch")
