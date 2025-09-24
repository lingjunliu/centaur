
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_clamp_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensor with min and max
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

    # Case 2: Integer tensor with min as tensor
    input_tensor = torch.randint(-5, 5, (3, 3)).numpy()
    min_tensor = torch.tensor([-1, -2, -3]).float()
    max_val = 2
    input_dict = {
        "input": input_tensor,
        "min": min_tensor.numpy(),
        "max": torch.tensor(max_val).numpy(),
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Tensor with only min
    input_tensor = torch.randn(2, 2, 2).numpy()
    min_val = 0.2
    input_dict = {
        "input": input_tensor,
        "min": min_val,
        "max": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Tensor with only max
    input_tensor = torch.randn(5).numpy()
    max_val = 1.0
    input_dict = {
        "input": input_tensor,
        "min": None,
        "max": max_val,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Tensor with min > max
    input_tensor = torch.randn(3, 4).numpy()
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
generated_inputs["torch.clamp_4"] = torch_clamp_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.clamp_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.clamp_4'.")

check_valid('torch.clamp', generated_inputs['torch.clamp_4'], lib="torch")
