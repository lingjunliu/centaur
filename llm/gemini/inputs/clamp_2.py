
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_clamp_inputs():
    list_of_inputs = []

    # Case 1: Basic case with float tensor, float min/max
    input_tensor = torch.randn(3, 4).numpy()
    min_tensor = torch.tensor(-0.5).numpy()
    max_value = 0.5
    input_dict = {
        "input": input_tensor,
        "min": min_tensor,
        "max": max_value,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Integer tensor, int min/max
    input_tensor = torch.randint(-5, 5, (2, 2), dtype=torch.int32).numpy()
    min_tensor = torch.tensor(-1).numpy()
    max_value = 2.0
    input_dict = {
        "input": input_tensor,
        "min": min_tensor,
        "max": max_value,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 1D tensor
    input_tensor = torch.randn(5).numpy()
    min_tensor = torch.tensor(0.0).numpy()
    max_value = 1.0
    input_dict = {
        "input": input_tensor,
        "min": min_tensor,
        "max": max_value,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 3D tensor
    input_tensor = torch.randn(2, 3, 4).numpy()
    min_tensor = torch.tensor(-1.0).numpy()
    max_value = 0.0
    input_dict = {
        "input": input_tensor,
        "min": min_tensor,
        "max": max_value,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: min > max
    input_tensor = torch.randn(2, 2).numpy()
    min_tensor = torch.tensor(1.0).numpy()
    max_value = -1.0
    input_dict = {
        "input": input_tensor,
        "min": min_tensor,
        "max": max_value,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Different min/max values
    input_tensor = torch.randn(3, 3).numpy()
    min_tensor = torch.tensor(-2.0).numpy()
    max_value = 3.0
    input_dict = {
        "input": input_tensor,
        "min": min_tensor,
        "max": max_value,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.clamp_2"] = torch_clamp_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.clamp_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.clamp_2'.")

check_valid('torch.clamp', generated_inputs['torch.clamp_2'], lib="torch")
