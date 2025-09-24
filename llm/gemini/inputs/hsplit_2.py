
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def hsplit_inputs():
    list_of_inputs = []

    # Case 1: 2D tensor, integer sections
    t = np.arange(16.0).reshape(4, 4)
    input_dict = {
        "input": t,
        "indices_or_sections": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 2D tensor, list of indices
    t = np.arange(16.0).reshape(4, 4)
    input_dict = {
        "input": t,
        "indices_or_sections": [1, 3]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 1D tensor, integer sections - Adjusted for divisibility
    t = np.arange(12.0)
    input_dict = {
        "input": t,
        "indices_or_sections": 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 1D tensor, list of indices
    t = np.arange(12.0)
    input_dict = {
        "input": t,
        "indices_or_sections": [2, 5, 8]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5:  2D tensor, integer sections, evenly divisible
    t = np.arange(10.0).reshape(2, 5)
    input_dict = {
        "input": t,
        "indices_or_sections": 5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.hsplit_2"] = hsplit_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.hsplit_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.hsplit_2'.")

check_valid('torch.hsplit', generated_inputs['torch.hsplit_2'], lib="torch")
