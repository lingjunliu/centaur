
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def PReLU_inputs():
    list_of_inputs = []

    # Input 1: Basic case with default parameters and 1D input
    input1 = torch.randn(5).numpy()
    input_dict1 = {
        "num_parameters": 1,
        "init": 0.25,
        "dtype": None,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D input with specific init value
    input2 = torch.randn(2, 3).numpy()
    input_dict2 = {
        "num_parameters": 1,
        "init": -0.5,
        "dtype": None,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D input with num_parameters equal to the number of channels
    input3 = torch.randn(2, 4, 5).numpy()
    input_dict3 = {
        "num_parameters": 4,
        "init": 0.1,
        "dtype": None,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 4D input with a different dtype
    input4 = torch.randn(1, 3, 4, 4, dtype=torch.float64).numpy()
    input_dict4 = {
        "num_parameters": 3,
        "init": 0.75,
        "dtype": torch.float64,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 1D input with negative values
    input5 = torch.randn(5).numpy() * -1
    input_dict5 = {
        "num_parameters": 1,
        "init": 0.25,
        "dtype": None,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Empty tensor
    input6 = torch.randn(0).numpy()
    input_dict6 = {
        "num_parameters": 1,
        "init": 0.25,
        "dtype": None,
        "input": input6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Tensor with only zeros
    input7 = torch.zeros(2, 3).numpy()
    input_dict7 = {
        "num_parameters": 1,
        "init": 0.25,
        "dtype": None,
        "input": input7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.PReLU"] = PReLU_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

check_valid('torch.nn.PReLU', generated_inputs['torch.nn.PReLU'], lib="torch")
