
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def tensordot_inputs():
    list_of_inputs = []

    # Case 1: Integer tensors, dims as integer
    a = np.arange(24).reshape((2, 3, 4))
    b = np.arange(12).reshape((3, 4))
    dims = 2
    input_dict = {"a": a, "b": b, "dims": dims, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Float tensors, dims as tuple of lists
    a = np.random.rand(3, 4, 5)
    b = np.random.rand(4, 5, 6)
    dims = ([1, 2], [0, 1])
    input_dict = {"a": a, "b": b, "dims": dims, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Different shapes, dims as a single list of lists
    a = np.random.rand(2, 3, 4, 5)
    b = np.random.rand(5, 4, 6)
    dims = [[3, 2], [0, 1]]
    input_dict = {"a": a, "b": b, "dims": dims, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Smaller tensors
    a = np.random.rand(2, 2)
    b = np.random.rand(2, 2)
    dims = 1
    input_dict = {"a": a, "b": b, "dims": dims, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: More dimensions
    a = np.random.rand(2, 3, 4, 5, 6)
    b = np.random.rand(6, 5, 4, 7)
    dims = ([4, 3, 2], [0, 1, 2])
    input_dict = {"a": a, "b": b, "dims": dims, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.tensordot_3"] = tensordot_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.tensordot_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.tensordot_3'.")

check_valid('torch.tensordot', generated_inputs['torch.tensordot_3'], lib="torch")
