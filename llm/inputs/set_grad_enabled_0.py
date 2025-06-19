
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def set_grad_enabled_inputs():
    list_of_inputs = []

    # Input 1: Enable grad
    input_dict = {"mode": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Disable grad
    input_dict = {"mode": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Enable grad
    input_dict = {"mode": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Disable grad
    input_dict = {"mode": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Enable Grad
    input_dict = {"mode": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.set_grad_enabled"] = set_grad_enabled_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.set_grad_enabled' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.set_grad_enabled'.")

check_valid('torch.set_grad_enabled', generated_inputs['torch.set_grad_enabled'], lib="torch", suffix=0)
