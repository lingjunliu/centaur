
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def empty_strided_inputs():
    list_of_inputs = []

    # Input 1
    size = (2, 3)
    stride = (3, 1)
    dtype = torch.float32
    layout = 'strided'
    pin_memory = False
    requires_grad = False

    input_dict = {
        "size": size,
        "stride": stride,
        "dtype": dtype,
        "layout": layout,
        "pin_memory": pin_memory,
        "requires_grad": requires_grad,
        "args": [size, stride]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    size = (5,)
    stride = (1,)
    dtype = torch.int64
    layout = 'strided'
    pin_memory = True
    requires_grad = True

    input_dict = {
        "size": size,
        "stride": stride,
        "dtype": dtype,
        "layout": layout,
        "pin_memory": pin_memory,
        "requires_grad": requires_grad,
        "args": [size, stride]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    size = (2, 2, 2)
    stride = (4, 2, 1)
    dtype = torch.float64
    layout = 'strided'
    pin_memory = False
    requires_grad = False

    input_dict = {
        "size": size,
        "stride": stride,
        "dtype": dtype,
        "layout": layout,
        "pin_memory": pin_memory,
        "requires_grad": requires_grad,
        "args": [size, stride]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    size = (1, 4, 1)
    stride = (4, 1, 4)
    dtype = torch.uint8
    layout = 'strided'
    pin_memory = True
    requires_grad = True

    input_dict = {
        "size": size,
        "stride": stride,
        "dtype": dtype,
        "layout": layout,
        "pin_memory": pin_memory,
        "requires_grad": requires_grad,
        "args": [size, stride]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    size = (2, 3, 4, 5)
    stride = (60, 20, 5, 1)
    dtype = torch.bool
    layout = 'strided'
    pin_memory = False
    requires_grad = False

    input_dict = {
        "size": size,
        "stride": stride,
        "dtype": dtype,
        "layout": layout,
        "pin_memory": pin_memory,
        "requires_grad": requires_grad,
        "args": [size, stride]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    size = (1,)
    stride = (1,)
    dtype = torch.complex64
    layout = 'strided'
    pin_memory = True
    requires_grad = True

    input_dict = {
        "size": size,
        "stride": stride,
        "dtype": dtype,
        "layout": layout,
        "pin_memory": pin_memory,
        "requires_grad": requires_grad,
        "args": [size, stride]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    size = (2, 1)
    stride = (1, 2)
    dtype = torch.int8
    layout = 'strided'
    pin_memory = False
    requires_grad = False

    input_dict = {
        "size": size,
        "stride": stride,
        "dtype": dtype,
        "layout": layout,
        "pin_memory": pin_memory,
        "requires_grad": requires_grad,
        "args": [size, stride]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8
    size = (3,3)
    stride = (3,1)
    dtype = torch.float16
    layout = 'strided'
    pin_memory = True
    requires_grad = True

    input_dict = {
        "size": size,
        "stride": stride,
        "dtype": dtype,
        "layout": layout,
        "pin_memory": pin_memory,
        "requires_grad": requires_grad,
        "args": [size, stride]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9
    size = (4, )
    stride = (1, )
    dtype = torch.uint16
    layout = 'strided'
    pin_memory = False
    requires_grad = False

    input_dict = {
        "size": size,
        "stride": stride,
        "dtype": dtype,
        "layout": layout,
        "pin_memory": pin_memory,
        "requires_grad": requires_grad,
        "args": [size, stride]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    size = (1, 2, 3)
    stride = (6, 3, 1)
    dtype = torch.int32
    layout = 'strided'
    pin_memory = True
    requires_grad = True

    input_dict = {
        "size": size,
        "stride": stride,
        "dtype": dtype,
        "layout": layout,
        "pin_memory": pin_memory,
        "requires_grad": requires_grad,
        "args": [size, stride]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.empty_strided"] = empty_strided_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.empty_strided' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.empty_strided'.")

check_valid('torch.empty_strided', generated_inputs['torch.empty_strided'], lib="torch", suffix=0)
