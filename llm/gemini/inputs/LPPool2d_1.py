
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def LPPool2d_inputs():
    list_of_inputs = []

    input1 = torch.randn(2, 3, 32, 32).numpy()
    norm_type1 = 2.0
    kernel_size1 = 3
    stride1 = 2
    ceil_mode1 = False
    input_dict1 = {
        "norm_type": norm_type1,
        "kernel_size": kernel_size1,
        "stride": stride1,
        "ceil_mode": ceil_mode1,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(1, 16, 64, 64).numpy()
    norm_type2 = 1.5
    kernel_size2 = (5, 5)
    stride2 = (3, 3)
    ceil_mode2 = True
    input_dict2 = {
        "norm_type": norm_type2,
        "kernel_size": kernel_size2,
        "stride": stride2,
        "ceil_mode": ceil_mode2,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(4, 8, 16, 16).numpy()
    norm_type3 = 3.0
    kernel_size3 = 2
    stride3 = None
    ceil_mode3 = False
    input_dict3 = {
        "norm_type": norm_type3,
        "kernel_size": kernel_size3,
        "stride": stride3,
        "ceil_mode": ceil_mode3,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(3, 32, 32).numpy()
    norm_type4 = 1.0
    kernel_size4 = (3, 2)
    stride4 = (2, 1)
    ceil_mode4 = True
    input_dict4 = {
        "norm_type": norm_type4,
        "kernel_size": kernel_size4,
        "stride": stride4,
        "ceil_mode": ceil_mode4,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(1, 4, 28, 28).numpy()
    norm_type5 = float('inf')
    kernel_size5 = 7
    stride5 = 1
    ceil_mode5 = False
    input_dict5 = {
        "norm_type": norm_type5,
        "kernel_size": kernel_size5,
        "stride": stride5,
        "ceil_mode": ceil_mode5,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = torch.randn(1, 1, 10, 10).numpy()
    norm_type6 = 0.5
    kernel_size6 = 3
    stride6 = 1
    ceil_mode6 = True
    input_dict6 = {
        "norm_type": norm_type6,
        "kernel_size": kernel_size6,
        "stride": stride6,
        "ceil_mode": ceil_mode6,
        "input": input6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.LPPool2d_1"] = LPPool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

check_valid('torch.nn.LPPool2d', generated_inputs['torch.nn.LPPool2d_1'], lib="torch")
