
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def FractionalMaxPool3d_inputs():
    list_of_inputs = []

    input1 = torch.randn(20, 16, 50, 32, 16).numpy()
    kernel_size1 = (3, 3, 3)
    output_size1 = (13, 12, 11)
    output_ratio1 = None
    return_indices1 = False

    input_dict1 = {
        "input": input1,
        "kernel_size": kernel_size1,
        "output_size": output_size1,
        "output_ratio": output_ratio1,
        "return_indices": return_indices1
    }

    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 3, 20, 20, 20).numpy()
    kernel_size2 = 2
    output_size2 = None
    output_ratio2 = (0.5, 0.5, 0.5)
    return_indices2 = True

    input_dict2 = {
        "input": input2,
        "kernel_size": kernel_size2,
        "output_size": output_size2,
        "output_ratio": output_ratio2,
        "return_indices": return_indices2
    }

    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(1, 1, 10, 10, 10).numpy()
    kernel_size3 = (2, 2, 2)
    output_size3 = (5, 5, 5)
    output_ratio3 = None
    return_indices3 = False

    input_dict3 = {
        "input": input3,
        "kernel_size": kernel_size3,
        "output_size": output_size3,
        "output_ratio": output_ratio3,
        "return_indices": return_indices3
    }

    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = torch.randn(5, 4, 30, 25, 20).numpy()
    kernel_size4 = 4
    output_size4 = None
    output_ratio4 = (0.6, 0.7, 0.8)
    return_indices4 = True
    
    input_dict4 = {
        "input": input4,
        "kernel_size": kernel_size4,
        "output_size": output_size4,
        "output_ratio": output_ratio4,
        "return_indices": return_indices4
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(10, 8, 64, 64, 64).numpy()
    kernel_size5 = (8, 8, 8)
    output_size5 = (32, 32, 32)
    output_ratio5 = None
    return_indices5 = False

    input_dict5 = {
        "input": input5,
        "kernel_size": kernel_size5,
        "output_size": output_size5,
        "output_ratio": output_ratio5,
        "return_indices": return_indices5
    }

    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.FractionalMaxPool3d_2"] = FractionalMaxPool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.FractionalMaxPool3d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.FractionalMaxPool3d_2'.")

check_valid('torch.nn.FractionalMaxPool3d', generated_inputs['torch.nn.FractionalMaxPool3d_2'], lib="torch")
