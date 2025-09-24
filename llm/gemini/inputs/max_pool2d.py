
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def max_pool2d_inputs():
    list_of_inputs = []

    input_float = torch.randn(1, 3, 32, 32).numpy()
    input_dict = {
        "input": input_float,
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "ceil_mode": False,
        "return_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_int = torch.randint(0, 10, (1, 1, 16, 16)).numpy()
    input_dict = {
        "input": input_int,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "dilation": 1,
        "ceil_mode": False,
        "return_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_neg = torch.randn(2, 4, 28, 28) * -1.0
    input_neg = input_neg.numpy()
    input_dict = {
        "input": input_neg,
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "ceil_mode": True,
        "return_indices": False #return_indices can only be true for CUDA tensors
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_no_batch = torch.randn(3, 16, 16).numpy()
    input_dict = {
        "input": input_no_batch,
        "kernel_size": 4,
        "stride": 4,
        "padding": 0,
        "dilation": 1,
        "ceil_mode": False,
        "return_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_large = torch.randn(4, 8, 64, 64).numpy()
    input_dict = {
        "input": input_large,
        "kernel_size": 8,
        "stride": 8,
        "padding": 0,
        "dilation": 1,
        "ceil_mode": False,
        "return_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_non_square = torch.randn(1, 1, 32, 64).numpy()
    input_dict = {
        "input": input_non_square,
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "ceil_mode": False,
        "return_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_padding = torch.randn(1, 1, 16, 16).numpy()
    input_dict = {
        "input": input_padding,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1, #Modified padding to 1
        "dilation": 1,
        "ceil_mode": False,
        "return_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dilation = torch.randn(1, 1, 16, 16).numpy()
    input_dict = {
        "input": input_dilation,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "dilation": 2,
        "ceil_mode": False,
        "return_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = max_pool2d_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('max_pool2d', generated_inputs)
