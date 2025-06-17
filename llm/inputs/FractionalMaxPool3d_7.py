
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def FractionalMaxPool3d_inputs():
    list_of_inputs = []

    input1 = torch.randn(20, 16, 50, 32, 16).numpy()
    input_dict1 = {
        "kernel_size": 3,
        "output_size": (13, 12, 11),
        "output_ratio": None,
        "return_indices": False,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(20, 16, 50, 32, 16).numpy()
    input_dict2 = {
        "kernel_size": 3,
        "output_size": None,
        "output_ratio": (0.5, 0.5, 0.5),
        "return_indices": False,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(1, 1, 20, 20, 20).numpy()
    input_dict3 = {
        "kernel_size": 2,
        "output_size": (10, 10, 10),
        "output_ratio": None,
        "return_indices": True,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(1, 1, 20, 20, 20).numpy()
    input_dict4 = {
        "kernel_size": (2, 3, 4),
        "output_size": None,
        "output_ratio": (0.6, 0.7, 0.8),
        "return_indices": True,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(2, 3, 25, 25, 25).numpy()
    input_dict5 = {
        "kernel_size": 5,
        "output_size": (7, 8, 9),
        "output_ratio": None,
        "return_indices": False,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(2, 3, 25, 25, 25).numpy()
    input_dict6 = {
        "kernel_size": (5, 4, 3),
        "output_size": None,
        "output_ratio": (0.3, 0.4, 0.5),
        "return_indices": False,
        "input": input6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.randn(1, 1, 15, 15, 15).numpy()
    input_dict7 = {
        "kernel_size": 3,
        "output_size": 5,
        "output_ratio": None,
        "return_indices": True,
        "input": input7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs["torch.nn.FractionalMaxPool3d_7"] = FractionalMaxPool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.FractionalMaxPool3d_7' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.FractionalMaxPool3d_7'.")

check_valid('torch.nn.FractionalMaxPool3d', generated_inputs['torch.nn.FractionalMaxPool3d_7'], lib="torch")
