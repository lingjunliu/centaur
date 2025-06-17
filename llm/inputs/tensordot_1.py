
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_tensordot_inputs():
    list_of_inputs = []

    # Case 1: Basic case with dims=2
    a = np.arange(60.).reshape(3, 4, 5)
    b = np.arange(24.).reshape(4, 3, 2)
    dims = 2
    input_dict = {"a": a, "b": b, "dims": dims, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: dims as a tuple of lists
    a = np.arange(60.).reshape(3, 4, 5)
    b = np.arange(24.).reshape(4, 3, 2)
    dims = ([1, 0], [0, 1])
    input_dict = {"a": a, "b": b, "dims": dims, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Different shapes and dims=1.  Make sure contracted dimensions match.
    a = np.arange(24.).reshape(2, 3, 4)
    b = np.arange(20.).reshape(5, 4) # shape[1] = 4 matches a.shape[2] and b is 2D
    dims = 1
    input_dict = {"a": a, "b": b, "dims": dims, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: dims as a list of lists with more dimensions. Make sure contracted dims match
    a = np.random.rand(3, 5, 4, 6)
    b = np.random.rand(6, 4, 5, 3)
    dims = ([3, 2, 1], [0, 1, 2])
    input_dict = {"a": a, "b": b, "dims": dims, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: dims as an integer, smaller tensors.  Make sure contracted dimensions match
    a = np.arange(12.).reshape(3, 4)
    b = np.arange(20.).reshape(4, 5)
    dims = 1
    input_dict = {"a": a, "b": b, "dims": dims, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.tensordot_1"] = torch_tensordot_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.tensordot_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.tensordot_1'.")

check_valid('torch.tensordot', generated_inputs['torch.tensordot_1'], lib="torch")
