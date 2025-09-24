
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def moveaxis_inputs():
    list_of_inputs = []

    # Example 1: 3D tensor, move axis 1 to axis 0
    t = torch.randn(3, 2, 1).numpy()
    input_dict = {
        "input": t,
        "source": 1,
        "destination": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: 4D tensor, move axis 2 to axis 3
    t = torch.randn(2, 3, 4, 5).numpy()
    input_dict = {
        "input": t,
        "source": 2,
        "destination": 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: 2D tensor, move axis 0 to axis 1
    t = torch.randn(5, 7).numpy()
    input_dict = {
        "input": t,
        "source": 0,
        "destination": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: 5D tensor, move axis 4 to axis 0
    t = torch.randn(1, 2, 3, 4, 5).numpy()
    input_dict = {
        "input": t,
        "source": 4,
        "destination": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 5: 3D tensor, move axis 0 to axis 2
    t = torch.randn(3, 2, 1).numpy()
    input_dict = {
        "input": t,
        "source": 0,
        "destination": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs["torch.moveaxis_1"] = moveaxis_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.moveaxis_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.moveaxis_1'.")

check_valid('torch.moveaxis', generated_inputs['torch.moveaxis_1'], lib="torch")
