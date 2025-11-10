
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy

def is_grad_enabled_inputs():
    list_of_inputs = []
    
    # Input 1
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.is_grad_enabled"] = is_grad_enabled_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.is_grad_enabled' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.is_grad_enabled'.")


check_valid('torch.is_grad_enabled', generated_inputs['torch.is_grad_enabled'], lib="torch", suffix=0)
