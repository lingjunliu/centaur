
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def empty_strided_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    size = (2, 3)
    stride = (3, 1)
    dtype = torch.float32
    input_dict = {
        "size": size,
        "stride": stride,
        "dtype": dtype,
        "layout": torch.strided,
        "pin_memory": False,
        "requires_grad": False,
        "args": (size, stride),
        "kwargs": {'dtype':dtype, 'layout': torch.strided, 'pin_memory': False, 'requires_grad': False}
    }
    list_of_inputs.append(input_dict)

    # Input 2: Int tensor with different strides
    size = (4, 5)
    stride = (5, 1)
    dtype = torch.int64
    input_dict = {
        "size": size,
        "stride": stride,
        "dtype": dtype,
        "layout": torch.strided,
        "pin_memory": False,
        "requires_grad": False,
        "args": (size, stride),
        "kwargs": {'dtype':dtype, 'layout': torch.strided, 'pin_memory': False, 'requires_grad': False}
    }
    list_of_inputs.append(input_dict)

    # Input 3: Complex tensor
    size = (2, 2)
    stride = (2, 1)
    dtype = torch.complex64
    input_dict = {
        "size": size,
        "stride": stride,
        "dtype": dtype,
        "layout": torch.strided,
        "pin_memory": False,
        "requires_grad": False,
        "args": (size, stride),
        "kwargs": {'dtype':dtype, 'layout': torch.strided, 'pin_memory': False, 'requires_grad': False}
    }
    list_of_inputs.append(input_dict)

    # Input 4: Bool tensor
    size = (3, 3)
    stride = (3, 1)
    dtype = torch.bool
    input_dict = {
        "size": size,
        "stride": stride,
        "dtype": dtype,
        "layout": torch.strided,
        "pin_memory": False,
        "requires_grad": False,
        "args": (size, stride),
        "kwargs": {'dtype':dtype, 'layout': torch.strided, 'pin_memory': False, 'requires_grad': False}
    }
    list_of_inputs.append(input_dict)

    # Input 5: High dimensional tensor
    size = (2, 3, 4)
    stride = (12, 4, 1)
    dtype = torch.float64
    input_dict = {
        "size": size,
        "stride": stride,
        "dtype": dtype,
        "layout": torch.strided,
        "pin_memory": False,
        "requires_grad": False,
        "args": (size, stride),
        "kwargs": {'dtype':dtype, 'layout': torch.strided, 'pin_memory': False, 'requires_grad': False}
    }
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.empty_strided"] = empty_strided_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.empty_strided' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.empty_strided'.")

check_valid('torch.empty_strided', generated_inputs['torch.empty_strided'], lib="torch")
