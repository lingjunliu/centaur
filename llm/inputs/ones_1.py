
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_ones_inputs():
    list_of_inputs = []

    # Example 1: Basic usage with different sizes
    input_dict = {"size": (2, 3)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"size": [5]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"size": (2, 3, 4)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 2: Using dtype
    input_dict = {"size": (2, 2), "dtype": torch.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.ones_1"] = torch_ones_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.ones_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.ones_1'.")

check_valid('torch.ones', generated_inputs['torch.ones_1'], lib="torch")
