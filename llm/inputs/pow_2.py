
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_pow_inputs():
    list_of_inputs = []

    # Case 1: input is a tensor, exponent is a float
    input_tensor = torch.randn(4).numpy()
    exponent = 2.0
    input_dict = {
        "input": input_tensor,
        "exponent": exponent,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: input is a tensor, exponent is a tensor of the same shape
    input_tensor = torch.arange(1., 5.).numpy()
    exponent_tensor = torch.arange(1., 5.).numpy()
    input_dict = {
        "input": input_tensor,
        "exponent": exponent_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: input is a tensor, exponent is a tensor with broadcastable shapes
    input_tensor = torch.randn(2, 3).numpy()
    exponent_tensor = torch.tensor(2.0).numpy()
    input_dict = {
        "input": input_tensor,
        "exponent": exponent_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 4: input is a tensor with negative values, exponent is a float
    input_tensor = torch.randn(4) * -1.0
    input_tensor = input_tensor.numpy()
    exponent = 3.0
    input_dict = {
        "input": input_tensor,
        "exponent": exponent,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: input is a tensor, exponent is a tensor with different dtype (int)
    input_tensor = torch.arange(1., 5.).numpy()
    exponent_tensor = torch.arange(1, 5).numpy()
    input_dict = {
        "input": input_tensor,
        "exponent": exponent_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Scalar base, exponent is a tensor
    base = 2.0
    exponent_tensor = torch.arange(1., 5.).numpy()
    input_dict = {
        "input": base,
        "exponent": exponent_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: input is a tensor, exponent is a float (negative)
    input_tensor = torch.randn(4).numpy()
    exponent = -2.0
    input_dict = {
        "input": input_tensor,
        "exponent": exponent,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 8: input is a tensor (int), exponent is a float
    input_tensor = torch.arange(1, 5).numpy()
    exponent = 2.0
    input_dict = {
        "input": input_tensor,
        "exponent": exponent,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.pow_2"] = torch_pow_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.pow_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.pow_2'.")

check_valid('torch.pow', generated_inputs['torch.pow_2'], lib="torch")
