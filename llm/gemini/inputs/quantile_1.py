
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def quantile_inputs():
    list_of_inputs = []

    # Case 1: Basic example with a 2D tensor, q as float
    input1 = torch.randn(2, 3).numpy()
    q1 = 0.5
    input_dict1 = {
        "input": input1,
        "q": q1,
        "dim": 1,
        "keepdim": True,
        "interpolation": "linear",
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: q as a 1D tensor
    input2 = torch.randn(4, 5).numpy()
    q2 = torch.tensor([0.25, 0.5, 0.75]).numpy()
    input_dict2 = {
        "input": input2,
        "q": q2,
        "dim": 0,
        "keepdim": False,
        "interpolation": "linear",
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: Different interpolation method, 1D input tensor
    input3 = torch.arange(5.).numpy()
    q3 = 0.6
    input_dict3 = {
        "input": input3,
        "q": q3,
        "dim": None,
        "keepdim": False,
        "interpolation": "nearest",
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: Negative values in input, different dim
    input4 = torch.randn(3, 4, 5).numpy()
    q4 = 0.3
    input_dict4 = {
        "input": input4,
        "q": q4,
        "dim": 2,
        "keepdim": True,
        "interpolation": "higher",
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Case 5: Flattened input (dim=None), midpoint interpolation
    input5 = torch.randn(2, 2).numpy()
    q5 = 0.8
    input_dict5 = {
        "input": input5,
        "q": q5,
        "dim": None,
        "keepdim": False,
        "interpolation": "midpoint",
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.quantile_1"] = quantile_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.quantile_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.quantile_1'.")

check_valid('torch.quantile', generated_inputs['torch.quantile_1'], lib="torch")
