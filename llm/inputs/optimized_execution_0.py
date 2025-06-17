
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy

def optimized_execution_inputs():
    list_of_inputs = []

    list_of_inputs.append({"enabled": True})

    return list_of_inputs

generated_inputs["torch.jit.optimized_execution"] = optimized_execution_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.jit.optimized_execution' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.jit.optimized_execution'.")

check_valid('torch.jit.optimized_execution', generated_inputs['torch.jit.optimized_execution'], lib="torch")
