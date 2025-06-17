
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def mse_loss_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensors, reduction='mean' (default)
    input1 = torch.randn(3, 5).numpy()
    target1 = torch.randn(3, 5).numpy()
    input_dict1 = {
        "input": input1,
        "target": target1,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: Different sized float tensors, reduction='sum'
    input2 = torch.randn(2, 4, 6).numpy()
    target2 = torch.randn(2, 4, 6).numpy()
    input_dict2 = {
        "input": input2,
        "target": target2,
        "size_average": None,
        "reduce": None,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: 1D tensors (vectors), reduction='none'
    input3 = torch.randn(10).numpy()
    target3 = torch.randn(10).numpy()
    input_dict3 = {
        "input": input3,
        "target": target3,
        "size_average": None,
        "reduce": None,
        "reduction": 'none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: Int tensors, reduction='mean'
    input4 = torch.randint(0, 10, (4, 4)).float().numpy() # Convert to float
    target4 = torch.randint(0, 10, (4, 4)).float().numpy() # Convert to float
    input_dict4 = {
        "input": input4,
        "target": target4,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Case 5: Tensors with negative values, reduction='sum'
    input5 = torch.randn(2, 3, 2).numpy()
    target5 = torch.randn(2, 3, 2).numpy()
    input_dict5 = {
        "input": input5,
        "target": target5,
        "size_average": None,
        "reduce": None,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.mse_loss"] = mse_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.mse_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.mse_loss'.")

check_valid('torch.nn.functional.mse_loss', generated_inputs['torch.nn.functional.mse_loss'], lib="torch")
