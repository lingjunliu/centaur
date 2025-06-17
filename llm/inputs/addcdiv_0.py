
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def addcdiv_inputs():
    list_of_inputs = []

    # Test case 1: Basic float tensors
    input_tensor = torch.randn(2, 3).numpy()
    tensor1 = torch.randn(2, 3).numpy()
    tensor2 = torch.randn(2, 3).numpy()
    value = 0.5

    input_dict = {
        "input": input_tensor,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: Broadcasting
    input_tensor = torch.randn(1, 3).numpy()
    tensor1 = torch.randn(3, 1).numpy()
    tensor2 = torch.randn(1, 3).numpy()
    value = 0.1

    input_dict = {
        "input": input_tensor,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3:  float tensors instead of int
    input_tensor = torch.randn(2, 2).numpy()
    tensor1 = torch.randn(2, 2).numpy()
    tensor2 = torch.randn(2, 2).numpy()
    value = 2.0

    input_dict = {
        "input": input_tensor,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: Negative values
    input_tensor = torch.randn(2, 2).numpy()
    tensor1 = torch.randn(2, 2).numpy()
    tensor2 = torch.randn(2, 2).numpy()
    value = -0.5

    input_dict = {
        "input": input_tensor,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 5: Different shapes
    input_tensor = torch.randn(5).numpy()
    tensor1 = torch.randn(5).numpy()
    tensor2 = torch.randn(5).numpy()
    value = 1.2

    input_dict = {
        "input": input_tensor,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.addcdiv"] = addcdiv_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.addcdiv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.addcdiv'.")

check_valid('torch.addcdiv', generated_inputs['torch.addcdiv'], lib="torch")
