
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def normalize_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D tensor
    input1 = torch.randn(3, 5).numpy()
    input_dict1 = {
        "input": input1,
        "p": 2.0,
        "dim": 1,
        "eps": 1e-12,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 3D tensor, different p and dim
    input2 = torch.randn(2, 3, 4).numpy()
    input_dict2 = {
        "input": input2,
        "p": 1.0,
        "dim": 2,
        "eps": 1e-8,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 1D tensor
    input3 = torch.randn(10).numpy()
    input_dict3 = {
        "input": input3,
        "p": 2.0,
        "dim": 0,
        "eps": 1e-12,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Tensor with negative values
    input4 = torch.randn(4, 4) * -1.0
    input4 = input4.numpy()
    input_dict4 = {
        "input": input4,
        "p": 2.0,
        "dim": 1,
        "eps": 1e-12,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Different dim value
    input5 = torch.randn(2, 3, 4).numpy()
    input_dict5 = {
        "input": input5,
        "p": 2.0,
        "dim": 0,
        "eps": 1e-12,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Integer tensor
    input6 = torch.randint(0, 10, (2, 5)).numpy()
    input_dict6 = {
        "input": input6.astype(np.float32),
        "p": 2.0,
        "dim": 1,
        "eps": 1e-12,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: 4D tensor
    input7 = torch.randn(2, 3, 4, 5).numpy()
    input_dict7 = {
        "input": input7,
        "p": 2.0,
        "dim": 2,
        "eps": 1e-12,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.nn.functional.normalize"] = normalize_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.normalize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.normalize'.")

check_valid('torch.nn.functional.normalize', generated_inputs['torch.nn.functional.normalize'], lib="torch")
