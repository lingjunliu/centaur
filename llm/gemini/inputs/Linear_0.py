
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def Linear_inputs():
    list_of_inputs = []

    # Test case 1: Basic case with bias=True and float32
    input = torch.randn(128, 20).numpy()
    input_dict = {
        "in_features": 20,
        "out_features": 30,
        "bias": True,
        "dtype": torch.float32,
        "input": input,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: No bias and float32
    input = torch.randn(64, 15).numpy()
    input_dict = {
        "in_features": 15,
        "out_features": 25,
        "bias": False,
        "dtype": torch.float32,
        "input": input,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: int64 input, but cast to float32
    input = torch.randint(0, 10, (32, 10)).float().numpy()
    input_dict = {
        "in_features": 10,
        "out_features": 5,
        "bias": True,
        "dtype": torch.float32,
        "input": input,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: Different input dimensions (3D)
    input = torch.randn(10, 5, 8).numpy()
    input_dict = {
        "in_features": 8,
        "out_features": 12,
        "bias": True,
        "dtype": torch.float32,
        "input": input,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 5: Another basic case
    input = torch.randn(256, 40).numpy()
    input_dict = {
        "in_features": 40,
        "out_features": 60,
        "bias": False,
        "dtype": torch.float32,
        "input": input,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.Linear"] = Linear_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

check_valid('torch.nn.Linear', generated_inputs['torch.nn.Linear'], lib="torch")
