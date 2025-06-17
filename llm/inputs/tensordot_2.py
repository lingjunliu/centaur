
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def tensordot_inputs():
    list_of_inputs = []

    # Case 1: dims is an integer
    a = np.arange(60.).reshape(3, 4, 5)
    b = np.arange(40.).reshape(5, 2, 4)
    dims = 1
    input_dict = {
        "a": a,
        "b": b,
        "dims": dims,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: dims is a tuple of lists
    a = np.arange(60.).reshape(3, 4, 5)
    b = np.arange(40.).reshape(5, 4, 2)
    dims = ([2], [0])
    input_dict = {
        "a": a,
        "b": b,
        "dims": dims,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Different shapes and dims
    a = np.random.randn(3, 5, 4)
    b = np.random.randn(4, 3, 2)
    dims = ([2], [0])
    input_dict = {
        "a": a,
        "b": b,
        "dims": dims,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Scalar tensors
    a = np.array(5.0)
    b = np.array(2.0)
    dims = 0
    input_dict = {
        "a": a,
        "b": b,
        "dims": dims,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: One dimension tensors and dims = 1
    a = np.arange(5)
    b = np.arange(5)
    dims = 1
    input_dict = {
        "a": a,
        "b": b,
        "dims": dims,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.tensordot_2"] = tensordot_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.tensordot_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.tensordot_2'.")

check_valid('torch.tensordot', generated_inputs['torch.tensordot_2'], lib="torch")
