
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_clamp_inputs():
    list_of_inputs = []

    # Case 1: Basic case with float tensor, min, and max
    input_tensor = torch.randn(4).numpy()
    min_val = -0.5
    max_val = 0.5
    input_dict = {
        "input": input_tensor,
        "min": min_val,
        "max": max_val,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Integer tensor with different min and max
    input_tensor = torch.randint(-5, 5, (3, 3)).numpy()
    min_val = -2.0
    max_val = 3.0
    input_dict = {
        "input": input_tensor,
        "min": min_val,
        "max": max_val,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 2D tensor with only min
    input_tensor = torch.randn(2, 5).numpy()
    min_val = 0.0
    max_val = None
    input_dict = {
        "input": input_tensor,
        "min": min_val,
        "max": max_val,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 3D tensor with only max
    input_tensor = torch.randn(3, 2, 4).numpy()
    min_val = None
    max_val = 1.0
    input_dict = {
        "input": input_tensor,
        "min": min_val,
        "max": max_val,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Tensor with min > max
    input_tensor = torch.randn(5).numpy()
    min_val = 1.0
    max_val = -1.0
    input_dict = {
        "input": input_tensor,
        "min": min_val,
        "max": max_val,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.clamp_1"] = torch_clamp_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.clamp_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.clamp_1'.")

check_valid('torch.clamp', generated_inputs['torch.clamp_1'], lib="torch")
