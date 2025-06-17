
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def scatter_reduce_inputs():
    list_of_inputs = []

    # Case 1: Basic example with sum reduction
    input_dict = {
        "input": np.zeros((5, 3)).astype(np.float32),
        "dim": 0,
        "index": np.array([[0, 1, 2], [0, 1, 4], [0, 2, 3], [0, 3, 4], [1, 2, 4]]).astype(np.int64),
        "src": np.arange(15).reshape(5, 3).astype(np.float32),
        "reduce": "sum",
        "include_self": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Using min reduction with include_self=True
    input_dict = {
        "input": np.ones((3, 3)) * 100.0,
        "dim": 1,
        "index": np.array([[0, 1, 2], [0, 1, 0], [1, 2, 1]]).astype(np.int64),
        "src": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32),
        "reduce": "amin",
        "include_self": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Using amax reduction with negative values and different output_size
    input_dict = {
        "input": np.full((2, 4), -100.0).astype(np.float32),
        "dim": 0,
        "index": np.array([[0, 1, 0, 1], [1, 0, 1, 0]]).astype(np.int64),
        "src": np.array([[-1, -2, -3, -4], [-5, -6, -7, -8]]).astype(np.float32),
        "reduce": "amax",
        "include_self": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Integer tensors with mean reduction (requires float input)
    input_dict = {
        "input": np.zeros((3, 2)).astype(np.float32),
        "dim": 1,
        "index": np.array([[0, 1], [1, 0], [0, 1]]).astype(np.int64),
        "src": np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32),
        "reduce": "mean",
        "include_self": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Complex Tensor - Removed due to dtype issues
    # input_dict = {
    #     "input": np.zeros((2, 2), dtype=np.complex64),
    #     "dim": 1,
    #     "index": np.array([[0, 1], [1, 0]]).astype(np.int64),
    #     "src": (np.array([[1, 2], [3, 4]]) + 1j * np.array([[5, 6], [7, 8]])).astype(np.complex64),
    #     "reduce": "sum",
    #     "include_self": False
    # }
    # list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.scatter_reduce_3"] = scatter_reduce_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.scatter_reduce_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.scatter_reduce_3'.")

check_valid('torch.scatter_reduce', generated_inputs['torch.scatter_reduce_3'], lib="torch")
