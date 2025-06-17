
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def avg_pool2d_inputs():
    list_of_inputs = []

    input1 = torch.randn(1, 3, 32, 32).numpy()
    kernel_size1 = 2
    stride1 = 2
    padding1 = 0
    ceil_mode1 = False
    count_include_pad1 = True
    divisor_override1 = 1

    input_dict1 = {
        "input": input1,
        "kernel_size": kernel_size1,
        "stride": stride1,
        "padding": padding1,
        "ceil_mode": ceil_mode1,
        "count_include_pad": count_include_pad1,
        "divisor_override": divisor_override1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(1, 1, 16, 16).numpy()
    kernel_size2 = 3
    stride2 = 1
    padding2 = 1
    ceil_mode2 = True
    count_include_pad2 = False
    divisor_override2 = 2

    input_dict2 = {
        "input": input2,
        "kernel_size": kernel_size2,
        "stride": stride2,
        "padding": padding2,
        "ceil_mode": ceil_mode2,
        "count_include_pad": count_include_pad2,
        "divisor_override": divisor_override2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(2, 4, 64, 64).numpy()
    kernel_size3 = (2, 2)
    stride3 = (2, 2)
    padding3 = (1, 1)
    ceil_mode3 = False
    count_include_pad3 = True
    divisor_override3 = 3

    input_dict3 = {
        "input": input3,
        "kernel_size": kernel_size3,
        "stride": stride3,
        "padding": padding3,
        "ceil_mode": ceil_mode3,
        "count_include_pad": count_include_pad3,
        "divisor_override": divisor_override3,
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(1, 1, 8, 8).double().numpy()
    kernel_size4 = 2
    stride4 = 2
    padding4 = 0
    ceil_mode4 = False
    count_include_pad4 = True
    divisor_override4 = 1

    input_dict4 = {
        "input": input4,
        "kernel_size": kernel_size4,
        "stride": stride4,
        "padding": padding4,
        "ceil_mode": ceil_mode4,
        "count_include_pad": count_include_pad4,
        "divisor_override": divisor_override4,
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randn(1, 3, 32, 32).half().numpy()
    kernel_size5 = 2
    stride5 = 2
    padding5 = 0
    ceil_mode5 = False
    count_include_pad5 = True
    divisor_override5 = 1

    input_dict5 = {
        "input": input5,
        "kernel_size": kernel_size5,
        "stride": stride5,
        "padding": padding5,
        "ceil_mode": ceil_mode5,
        "count_include_pad": count_include_pad5,
        "divisor_override": divisor_override5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.nn.functional.avg_pool2d_1"] = avg_pool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.avg_pool2d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.avg_pool2d_1'.")

check_valid('torch.nn.functional.avg_pool2d', generated_inputs['torch.nn.functional.avg_pool2d_1'], lib="torch")
