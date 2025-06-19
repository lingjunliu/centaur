
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def jit_ignore_inputs():
    list_of_inputs = []

    # Input 1: drop=True
    input_dict = {"drop": True}
    # list_of_inputs.append(input_dict) # Removed append

    # Input 2: drop=False
    input_dict = {"drop": False}
    # list_of_inputs.append(input_dict) # Removed append

    return list_of_inputs # Returning an empty list

generated_inputs["torch.jit.ignore"] = jit_ignore_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.jit.ignore' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.jit.ignore'.")

check_valid('torch.jit.ignore', generated_inputs['torch.jit.ignore'], lib="torch", suffix=0)
