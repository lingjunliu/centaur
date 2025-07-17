
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
    dtype = np.float32

    input_dict = {
        "size": size,
        "stride": stride,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    size = (4, 5, 2)
    stride = (10, 2, 1)
    dtype = np.int64

    input_dict = {
        "size": size,
        "stride": stride,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    size = (1, 7)
    stride = (7, 1)
    dtype = np.bool_

    input_dict = {
        "size": size,
        "stride": stride,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    size = (2, 2, 2, 2)
    stride = (8, 4, 2, 1)
    dtype = np.float64

    input_dict = {
        "size": size,
        "stride": stride,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    size = (10,)
    stride = (1,)
    dtype = np.uint8

    input_dict = {
        "size": size,
        "stride": stride,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    size = (3, 1, 5)
    stride = (5, 5, 1)
    dtype = np.int32

    input_dict = {
        "size": size,
        "stride": stride,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    size = (1,)
    stride = (1,)
    dtype = np.float16

    input_dict = {
        "size": size,
        "stride": stride,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    size = (2, 4)
    stride = (4, 1)
    dtype = np.complex64

    input_dict = {
        "size": size,
        "stride": stride,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    size = (2, 2)
    stride = (2, 1)
    dtype = np.int8

    input_dict = {
        "size": size,
        "stride": stride,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    size = (5, 1)
    stride = (1, 1)
    dtype = np.uint16

    input_dict = {
        "size": size,
        "stride": stride,
        "dtype": dtype
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
