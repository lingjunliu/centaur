
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def sym_fresh_size_inputs():
    list_of_inputs = []

    # Example 1: Simple expression
    input_dict = {"args": ["x"]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Another simple expression
    input_dict = {"args": ["y"]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: Slightly more complex expression
    input_dict = {"args": ["a + b"]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: Numerical expression
    input_dict = {"args": ["10"]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5: Combination of variables and numbers
    input_dict = {"args": ["z + 5"]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.sym_fresh_size"] = sym_fresh_size_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sym_fresh_size' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sym_fresh_size'.")

check_valid('torch.sym_fresh_size', generated_inputs['torch.sym_fresh_size'], lib="torch")
