
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def quantize_per_tensor_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input1 = torch.randn(3, 4).numpy()
    scale1 = 0.5
    zero_point1 = 0
    dtype1 = torch.qint8

    input_dict1 = {
        "input": input1,
        "scale": scale1,
        "zero_point": zero_point1,
        "dtype": dtype1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Float tensor with negative values
    input2 = torch.randn(2, 2, 2).numpy()
    scale2 = 0.1
    zero_point2 = 5
    dtype2 = torch.quint8

    input_dict2 = {
        "input": input2,
        "scale": scale2,
        "zero_point": zero_point2,
        "dtype": dtype2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 1D tensor
    input3 = torch.randn(10).numpy()
    scale3 = 1.0
    zero_point3 = 2
    dtype3 = torch.quint8

    input_dict3 = {
        "input": input3,
        "scale": scale3,
        "zero_point": zero_point3,
        "dtype": dtype3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: Large tensor
    input4 = torch.randn(100, 100).numpy()
    scale4 = 0.01
    zero_point4 = 128
    dtype4 = torch.quint8
    
    input_dict4 = {
        "input": input4,
        "scale": scale4,
        "zero_point": zero_point4,
        "dtype": dtype4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Tensor with large scale
    input5 = torch.randn(5, 5).numpy()
    scale5 = 10.0
    zero_point5 = 0
    dtype5 = torch.qint8

    input_dict5 = {
        "input": input5,
        "scale": scale5,
        "zero_point": zero_point5,
        "dtype": dtype5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 4D Tensor
    input6 = torch.randn(1, 3, 224, 224).numpy()
    scale6 = 0.005
    zero_point6 = 0
    dtype6 = torch.qint8
    
    input_dict6 = {
        "input": input6,
        "scale": scale6,
        "zero_point": zero_point6,
        "dtype": dtype6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs = quantize_per_tensor_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('quantize_per_tensor', generated_inputs)
