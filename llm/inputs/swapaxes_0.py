
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def swapaxes_inputs():
    list_of_inputs = []

    # Example 1: 3D float tensor
    input_tensor = torch.randn(2, 3, 4).numpy()
    axis0 = 0
    axis1 = 1
    input_dict = {"input": input_tensor, "axis0": axis0, "axis1": axis1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: 4D int tensor with negative axes
    input_tensor = torch.randint(0, 10, (2, 3, 4, 5)).numpy()
    axis0 = 1
    axis1 = -1
    input_dict = {"input": input_tensor, "axis0": axis0, "axis1": axis1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: 2D complex tensor
    input_tensor = torch.randn(2, 3, dtype=torch.complex64).numpy()
    axis0 = 0
    axis1 = 1
    input_dict = {"input": input_tensor, "axis0": axis0, "axis1": axis1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: 1D tensor
    input_tensor = torch.arange(5).numpy()
    axis0 = 0
    axis1 = 0  # No effect, but valid
    input_dict = {"input": input_tensor, "axis0": axis0, "axis1": axis1}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 5: 5D tensor
    input_tensor = torch.randn(2, 3, 4, 5, 6).numpy()
    axis0 = 2
    axis1 = 4
    input_dict = {"input": input_tensor, "axis0": axis0, "axis1": axis1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 6: Bool tensor
    input_tensor = torch.tensor([[True, False], [False, True]]).numpy()
    axis0 = 0
    axis1 = 1
    input_dict = {"input": input_tensor, "axis0": axis0, "axis1": axis1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 7: 3D tensor with same axes
    input_tensor = torch.randn(2, 3, 4).numpy()
    axis0 = 1
    axis1 = 1
    input_dict = {"input": input_tensor, "axis0": axis0, "axis1": axis1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.swapaxes"] = swapaxes_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.swapaxes' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.swapaxes'.")

check_valid('torch.swapaxes', generated_inputs['torch.swapaxes'], lib="torch")
