
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_where_inputs():
    list_of_inputs = []

    # Case 1: Basic case with float tensors
    condition = (torch.randn(3, 2) > 0).numpy()
    input_tensor = torch.randn(3, 2).numpy()
    other_tensor = torch.randn(3, 2).numpy()
    input_dict = {
        "condition": condition,
        "input": input_tensor,
        "other": other_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Scalar input and other
    condition = (torch.randn(2, 2) > 0).numpy()
    input_scalar = torch.tensor(1.0).numpy()
    other_scalar = torch.tensor(0.0).numpy()
    input_dict = {
        "condition": condition,
        "input": input_scalar,
        "other": other_scalar,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Integer tensors
    condition = (torch.randint(0, 2, (4, 3)) > 0).numpy()
    input_tensor = torch.randint(-5, 5, (4, 3)).numpy()
    other_tensor = torch.randint(-5, 5, (4, 3)).numpy()
    input_dict = {
        "condition": condition,
        "input": input_tensor,
        "other": other_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Broadcasting condition
    condition = (torch.randn(3, 1) > 0).numpy()
    input_tensor = torch.randn(3, 2).numpy()
    other_tensor = torch.randn(3, 2).numpy()
    input_dict = {
        "condition": condition,
        "input": input_tensor,
        "other": other_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Broadcasting input/other (scalar)
    condition = (torch.randn(2, 2) > 0).numpy()
    input_scalar = torch.tensor(5.0).numpy()
    other_tensor = torch.randn(2, 2).numpy()
    input_dict = {
        "condition": condition,
        "input": input_scalar,
        "other": other_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.where_1"] = torch_where_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.where_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.where_1'.")

check_valid('torch.where', generated_inputs['torch.where_1'], lib="torch")
