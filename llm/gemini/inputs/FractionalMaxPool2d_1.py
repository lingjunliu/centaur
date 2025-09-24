
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def FractionalMaxPool2d_inputs():
    list_of_inputs = []

    input1 = torch.randn(20, 16, 50, 32).numpy()
    kernel_size1 = 3
    output_size1 = (13, 12)
    output_ratio1 = None
    return_indices1 = False
    input_dict1 = {
        "kernel_size": kernel_size1,
        "output_size": output_size1,
        "output_ratio": output_ratio1,
        "return_indices": return_indices1,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(20, 16, 50, 32).numpy()
    kernel_size2 = 3
    output_size2 = None
    output_ratio2 = (0.5, 0.5)
    return_indices2 = True
    input_dict2 = {
        "kernel_size": kernel_size2,
        "output_size": output_size2,
        "output_ratio": output_ratio2,
        "return_indices": return_indices2,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(1, 1, 20, 20).numpy()
    kernel_size3 = (2, 2)
    output_size3 = (10, 10)
    output_ratio3 = None
    return_indices3 = False
    input_dict3 = {
        "kernel_size": kernel_size3,
        "output_size": output_size3,
        "output_ratio": output_ratio3,
        "return_indices": return_indices3,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(1, 3, 25, 25).numpy()
    kernel_size4 = (3, 3)
    output_size4 = None
    output_ratio4 = (0.6, 0.6)
    return_indices4 = True
    input_dict4 = {
        "kernel_size": kernel_size4,
        "output_size": output_size4,
        "output_ratio": output_ratio4,
        "return_indices": return_indices4,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(2, 4, 30, 30).numpy()
    kernel_size5 = 4
    output_size5 = (15, 15)
    output_ratio5 = None
    return_indices5 = False
    input_dict5 = {
        "kernel_size": kernel_size5,
        "output_size": output_size5,
        "output_ratio": output_ratio5,
        "return_indices": return_indices5,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.FractionalMaxPool2d_1"] = FractionalMaxPool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

check_valid('torch.nn.FractionalMaxPool2d', generated_inputs['torch.nn.FractionalMaxPool2d_1'], lib="torch")
