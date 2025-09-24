
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def FractionalMaxPool3d_inputs():
    list_of_inputs = []

    input1 = torch.randn(2, 3, 20, 32, 16).numpy()
    kernel_size1 = 3
    output_size1 = (10, 16, 8)
    return_indices1 = False

    input_dict1 = {
        "kernel_size": kernel_size1,
        "output_size": output_size1,
        "output_ratio": None,
        "return_indices": return_indices1,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(1, 1, 15, 25, 10).numpy()
    kernel_size2 = (2, 3, 2)
    output_ratio2 = (0.6, 0.4, 0.7)
    return_indices2 = True
    input_dict2 = {
        "kernel_size": kernel_size2,
        "output_size": None,
        "output_ratio": output_ratio2,
        "return_indices": return_indices2,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(4, 5, 30, 40, 20).numpy()
    kernel_size3 = 4
    output_size3 = (7, 10, 5)
    return_indices3 = False
    input_dict3 = {
        "kernel_size": kernel_size3,
        "output_size": output_size3,
        "output_ratio": None,
        "return_indices": return_indices3,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(1, 2, 12, 18, 9).numpy()
    kernel_size4 = (3, 2, 1)
    output_ratio4 = (0.3, 0.8, 0.6)
    return_indices4 = True
    input_dict4 = {
        "kernel_size": kernel_size4,
        "output_size": None,
        "output_ratio": output_ratio4,
        "return_indices": return_indices4,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(3, 4, 25, 35, 15).numpy()
    kernel_size5 = 2
    output_size5 = (12, 17, 7)
    return_indices5 = False
    input_dict5 = {
        "kernel_size": kernel_size5,
        "output_size": output_size5,
        "output_ratio": None,
        "return_indices": return_indices5,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.FractionalMaxPool3d_1"] = FractionalMaxPool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.FractionalMaxPool3d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.FractionalMaxPool3d_1'.")

check_valid('torch.nn.FractionalMaxPool3d', generated_inputs['torch.nn.FractionalMaxPool3d_1'], lib="torch")
