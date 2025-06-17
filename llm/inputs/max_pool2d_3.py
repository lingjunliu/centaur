
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def max_pool2d_inputs():
    list_of_inputs = []

    input1 = torch.randn(1, 3, 32, 32).numpy()
    kernel_size1 = 3
    stride1 = (2, 2)
    padding1 = (1, 1)
    dilation1 = 1
    return_indices1 = False
    ceil_mode1 = False

    input_dict1 = {
        "input": input1,
        "kernel_size": kernel_size1,
        "stride": stride1,
        "padding": padding1,
        "dilation": dilation1,
        "return_indices": return_indices1,
        "ceil_mode": ceil_mode1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randint(0, 10, (2, 1, 16, 16)).numpy()
    kernel_size2 = 2
    stride2 = (1, 1)
    padding2 = (0, 0)
    dilation2 = 2
    return_indices2 = True
    ceil_mode2 = True

    input_dict2 = {
        "input": input2,
        "kernel_size": kernel_size2,
        "stride": stride2,
        "padding": padding2,
        "dilation": dilation2,
        "return_indices": return_indices2,
        "ceil_mode": ceil_mode2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(1, 4, 64, 64).numpy()
    kernel_size3 = (4, 4)
    stride3 = (4, 4)
    padding3 = (0, 0)
    dilation3 = 1
    return_indices3 = False
    ceil_mode3 = False

    input_dict3 = {
        "input": input3,
        "kernel_size": kernel_size3,
        "stride": stride3,
        "padding": padding3,
        "dilation": dilation3,
        "return_indices": return_indices3,
        "ceil_mode": ceil_mode3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(2, 3, 10, 10).numpy()
    kernel_size4 = 3
    stride4 = (2, 1)
    padding4 = (1, 0)
    dilation4 = 1
    return_indices4 = False
    ceil_mode4 = True

    input_dict4 = {
        "input": input4,
        "kernel_size": kernel_size4,
        "stride": stride4,
        "padding": padding4,
        "dilation": dilation4,
        "return_indices": return_indices4,
        "ceil_mode": ceil_mode4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(1, 1, 7, 7).numpy()
    kernel_size5 = 2
    stride5 = (2, 2)
    padding5 = (0, 0)
    dilation5 = 1
    return_indices5 = False
    ceil_mode5 = False

    input_dict5 = {
        "input": input5,
        "kernel_size": kernel_size5,
        "stride": stride5,
        "padding": padding5,
        "dilation": dilation5,
        "return_indices": return_indices5,
        "ceil_mode": ceil_mode5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.nn.functional.max_pool2d_3"] = max_pool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.max_pool2d_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.max_pool2d_3'.")

check_valid('torch.nn.functional.max_pool2d', generated_inputs['torch.nn.functional.max_pool2d_3'], lib="torch")
