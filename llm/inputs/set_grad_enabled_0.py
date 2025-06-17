
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy

def set_grad_enabled_inputs():
    list_of_inputs = []

    input_dict1 = {"mode": True}
    list_of_inputs.append(input_dict1)

    input_dict2 = {"mode": False}
    list_of_inputs.append(input_dict2)

    return list_of_inputs

generated_inputs["torch.set_grad_enabled"] = set_grad_enabled_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.set_grad_enabled' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.set_grad_enabled'.")

check_valid('torch.set_grad_enabled', generated_inputs['torch.set_grad_enabled'], lib="torch")
