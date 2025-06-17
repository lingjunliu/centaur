
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def tensordot_inputs():
    list_of_inputs = []

    # Case 1: Basic case with dims as an integer
    a = np.arange(60.).reshape(3, 4, 5)
    b = np.arange(60.).reshape(4, 3, 5)
    dims = ([1, 0], [0, 1])
    input_dict = {"a": a, "b": b, "dims": dims, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: dims as a tuple of lists
    a = np.random.randn(3, 5, 4, 6)
    b = np.random.randn(6, 4, 5, 3)
    dims = ([2, 1, 3], [1, 2, 0])
    input_dict = {"a": a, "b": b, "dims": dims, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Different shapes and integer dims
    a = np.random.randn(2, 3, 4)
    b = np.random.randn(4, 5, 6)
    dims = 1
    input_dict = {"a": a, "b": b, "dims": dims, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Different shapes and tuple of lists
    a = np.random.randn(3, 4, 5)
    b = np.random.randn(5, 2, 3)
    dims = ([2], [0])
    input_dict = {"a": a, "b": b, "dims": dims, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Simple matrix multiplication
    a = np.random.randn(5, 3)
    b = np.random.randn(3, 2)
    dims = 1
    input_dict = {"a": a, "b": b, "dims": dims, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.tensordot_4"] = tensordot_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.tensordot_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.tensordot_4'.")

check_valid('torch.tensordot', generated_inputs['torch.tensordot_4'], lib="torch")
