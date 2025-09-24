
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def moveaxis_inputs():
    list_of_inputs = []

    # Example 1: Basic case with positive indices
    t = torch.randn(3, 2, 1).numpy()
    input_dict = {
        "input": t,
        "source": (1,),
        "destination": (0,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Multiple axes moved
    t = torch.randn(3, 2, 1).numpy()
    input_dict = {
        "input": t,
        "source": (1, 2),
        "destination": (0, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: Negative indices
    t = torch.randn(3, 2, 4).numpy()
    input_dict = {
        "input": t,
        "source": (-1,),
        "destination": (0,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: Moving to the same position (no change)
    t = torch.randn(3, 2, 4).numpy()
    input_dict = {
        "input": t,
        "source": (0,),
        "destination": (0,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 5: Moving a single axis in a higher dimensional tensor
    t = torch.randn(2, 3, 4, 5).numpy()
    input_dict = {
        "input": t,
        "source": (2,),
        "destination": (0,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.moveaxis_2"] = moveaxis_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.moveaxis_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.moveaxis_2'.")

check_valid('torch.moveaxis', generated_inputs['torch.moveaxis_2'], lib="torch")
