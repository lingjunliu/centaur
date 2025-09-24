
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy as np

def torch_empty_inputs():
    list_of_inputs = []

    # Input 1
    size = (2, 3)
    out = torch.empty(2, 3)
    dtype = torch.int64
    layout = "strided"
    requires_grad = False
    pin_memory = False
    memory_format = "contiguous_format"

    input_dict = {
        "size": size,
        "out": out,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad,
        "pin_memory": pin_memory,
        "memory_format": memory_format
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    size = (5,)
    out = torch.empty(5)
    dtype = torch.float32
    layout = "strided"
    requires_grad = True
    pin_memory = False
    memory_format = "contiguous_format"

    input_dict = {
        "size": size,
        "out": out,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad,
        "pin_memory": pin_memory,
        "memory_format": memory_format
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    size = (2, 2, 2)
    out = torch.empty(2, 2, 2)
    dtype = torch.float64
    layout = "strided"
    requires_grad = False
    pin_memory = True
    memory_format = "contiguous_format"

    input_dict = {
        "size": size,
        "out": out,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad,
        "pin_memory": pin_memory,
        "memory_format": memory_format
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    size = (1, 5, 1, 5)
    out = torch.empty(1, 5, 1, 5)
    dtype = torch.int32
    layout = "strided"
    requires_grad = True
    pin_memory = True
    memory_format = "contiguous_format"

    input_dict = {
        "size": size,
        "out": out,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad,
        "pin_memory": pin_memory,
        "memory_format": memory_format
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5
    size = (10,)
    out = torch.empty(10)
    dtype = torch.uint8
    layout = "strided"
    requires_grad = False
    pin_memory = False
    memory_format = "contiguous_format"

    input_dict = {
        "size": size,
        "out": out,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad,
        "pin_memory": pin_memory,
        "memory_format": memory_format
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.empty_2"] = torch_empty_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.empty_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.empty_2'.")

check_valid('torch.empty', generated_inputs['torch.empty_2'], lib="torch", suffix=2)
