
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def FractionalMaxPool3d_inputs():
    list_of_inputs = []

    input1 = torch.randn(2, 3, 20, 32, 16).numpy()
    input_dict1 = {
        "kernel_size": 3,
        "output_size": (13, 12, 11),
        "output_ratio": None,
        "return_indices": False,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 3, 20, 32, 16).numpy()
    input_dict2 = {
        "kernel_size": 3,
        "output_size": None,
        "output_ratio": (0.5, 0.5, 0.5),
        "return_indices": True,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(1, 1, 16, 16, 16).numpy()
    input_dict3 = {
        "kernel_size": (2, 2, 2),
        "output_size": (8, 8, 8),
        "output_ratio": None,
        "return_indices": False,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(1, 1, 32, 32, 32).numpy()
    input_dict4 = {
        "kernel_size": (3, 3, 3),
        "output_size": None,
        "output_ratio": 0.25,
        "return_indices": True,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randn(4, 5, 64, 64, 64).numpy()
    input_dict5 = {
        "kernel_size": 4,
        "output_size": (16, 16, 16),
        "output_ratio": None,
        "return_indices": False,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(4, 5, 64, 64, 64).numpy()
    input_dict6 = {
        "kernel_size": 4,
        "output_size": None,
        "output_ratio": (0.25, 0.3, 0.4),
        "return_indices": True,
        "input": input6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.randn(1, 1, 10, 10, 10).numpy()
    input_dict7 = {
        "kernel_size": 2,
        "output_size": 5,
        "output_ratio": None,
        "return_indices": False,
        "input": input7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.FractionalMaxPool3d_5"] = FractionalMaxPool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.FractionalMaxPool3d_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.FractionalMaxPool3d_5'.")

check_valid('torch.nn.FractionalMaxPool3d', generated_inputs['torch.nn.FractionalMaxPool3d_5'], lib="torch")
