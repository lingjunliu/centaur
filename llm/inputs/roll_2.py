
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_roll_inputs():
    list_of_inputs = []

    # Example 1: 1D tensor, positive shift
    input_tensor = torch.tensor([1, 2, 3, 4, 5]).numpy()
    shifts = (2,)
    dims = (0,)
    input_dict = {"input": input_tensor, "shifts": shifts, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: 2D tensor, negative shift along one dimension
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).numpy()
    shifts = (-1,)
    dims = (0,)
    input_dict = {"input": input_tensor, "shifts": shifts, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: 2D tensor, shifts along both dimensions
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).numpy()
    shifts = (1, -1)
    dims = (0, 1)
    input_dict = {"input": input_tensor, "shifts": shifts, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: 3D tensor, shifts along all dimensions
    input_tensor = torch.randn(2, 3, 4).numpy()
    shifts = (1, -1, 2)
    dims = (0, 1, 2)
    input_dict = {"input": input_tensor, "shifts": shifts, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5: 1D tensor, zero shift
    input_tensor = torch.tensor([1, 2, 3, 4, 5]).numpy()
    shifts = (0,)
    dims = (0,)
    input_dict = {"input": input_tensor, "shifts": shifts, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 6: 2D tensor with different datatypes
    input_tensor = torch.tensor([[1.1, 2.2], [3.3, 4.4]]).numpy()
    shifts = (1,)
    dims = (0,)
    input_dict = {"input": input_tensor, "shifts": shifts, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 7: 2D tensor with larger shifts
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).numpy()
    shifts = (2, 2)
    dims = (0, 1)
    input_dict = {"input": input_tensor, "shifts": shifts, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.roll_2"] = torch_roll_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.roll_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.roll_2'.")

check_valid('torch.roll', generated_inputs['torch.roll_2'], lib="torch")
