
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def FractionalMaxPool3d_inputs():
    list_of_inputs = []

    # Case 1: Basic case with output_size
    input = torch.randn(2, 3, 20, 20, 20).numpy()
    kernel_size = 3
    output_size = (10, 10, 10)
    output_ratio = None
    return_indices = False
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "output_size": output_size,
        "output_ratio": output_ratio,
        "return_indices": return_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Basic case with output_ratio
    input = torch.randn(2, 3, 20, 20, 20).numpy()
    kernel_size = 3
    output_size = None
    output_ratio = (0.5, 0.5, 0.5)
    return_indices = False
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "output_size": output_size,
        "output_ratio": output_ratio,
        "return_indices": return_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Different kernel size
    input = torch.randn(2, 3, 20, 20, 20).numpy()
    kernel_size = (2, 3, 4)
    output_size = (10, 10, 10)
    output_ratio = None
    return_indices = False
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "output_size": output_size,
        "output_ratio": output_ratio,
        "return_indices": return_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Different output size
    input = torch.randn(2, 3, 20, 20, 20).numpy()
    kernel_size = 3
    output_size = (12, 15, 8)
    output_ratio = None
    return_indices = False
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "output_size": output_size,
        "output_ratio": output_ratio,
        "return_indices": return_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: return_indices = True
    input = torch.randn(2, 3, 20, 20, 20).numpy()
    kernel_size = 3
    output_size = (10, 10, 10)
    output_ratio = None
    return_indices = True
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "output_size": output_size,
        "output_ratio": output_ratio,
        "return_indices": return_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.FractionalMaxPool3d_3"] = FractionalMaxPool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.FractionalMaxPool3d_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.FractionalMaxPool3d_3'.")

check_valid('torch.nn.FractionalMaxPool3d', generated_inputs['torch.nn.FractionalMaxPool3d_3'], lib="torch")
