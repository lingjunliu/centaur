
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def uninitializedbuffer_inputs():
    list_of_inputs = []

    # Input 1
    size = (2, 3)
    dtype = np.dtype('float32')
    requires_grad = False
    pin_memory = False

    input_dict = {
        "size": size,
        "dtype": dtype,
        "requires_grad": requires_grad,
        "pin_memory": pin_memory,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    size = (5,)
    dtype = np.dtype('int64')
    requires_grad = True
    pin_memory = True

    input_dict = {
        "size": size,
        "dtype": dtype,
        "requires_grad": requires_grad,
        "pin_memory": pin_memory,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    size = (1, 4, 2)
    dtype = np.dtype('float16')
    requires_grad = False
    pin_memory = False

    input_dict = {
        "size": size,
        "dtype": dtype,
        "requires_grad": requires_grad,
        "pin_memory": pin_memory,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    size = (2, 2, 2, 2)
    dtype = np.dtype('uint8')
    requires_grad = True
    pin_memory = True

    input_dict = {
        "size": size,
        "dtype": dtype,
        "requires_grad": requires_grad,
        "pin_memory": pin_memory,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    size = (10,)
    dtype = np.dtype('bool')
    requires_grad = False
    pin_memory = False

    input_dict = {
        "size": size,
        "dtype": dtype,
        "requires_grad": requires_grad,
        "pin_memory": pin_memory,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    size = (3, 5, 7)
    dtype = np.dtype('complex64')
    requires_grad = True
    pin_memory = True

    input_dict = {
        "size": size,
        "dtype": dtype,
        "requires_grad": requires_grad,
        "pin_memory": pin_memory,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    size = (1,)
    dtype = np.dtype('float64')
    requires_grad = False
    pin_memory = False
    input_dict = {
        "size": size,
        "dtype": dtype,
        "requires_grad": requires_grad,
        "pin_memory": pin_memory,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    size = (8, 2)
    dtype = np.dtype('int32')
    requires_grad = True
    pin_memory = True

    input_dict = {
        "size": size,
        "dtype": dtype,
        "requires_grad": requires_grad,
        "pin_memory": pin_memory,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    size = (1, 1, 1, 1, 1)
    dtype = np.dtype('int8')
    requires_grad = False
    pin_memory = False

    input_dict = {
        "size": size,
        "dtype": dtype,
        "requires_grad": requires_grad,
        "pin_memory": pin_memory,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10
    size = (4, 4)
    dtype = np.dtype('complex128')
    requires_grad = True
    pin_memory = True

    input_dict = {
        "size": size,
        "dtype": dtype,
        "requires_grad": requires_grad,
        "pin_memory": pin_memory,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.UninitializedBuffer"] = uninitializedbuffer_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.UninitializedBuffer' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.UninitializedBuffer'.")

check_valid('torch.nn.UninitializedBuffer', generated_inputs['torch.nn.UninitializedBuffer'], lib="torch", suffix=0)
