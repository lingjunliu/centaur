
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def scatter_reduce_inputs():
    list_of_inputs = []

    # Example 1: Basic sum reduction
    input_dict = {
        "input": np.zeros((5,), dtype=np.float32),
        "dim": 0,
        "index": np.array([0, 1, 2, 0, 3], dtype=np.int64),
        "src": np.array([1, 2, 3, 4, 5], dtype=np.float32),
        "reduce": "sum",
        "include_self": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Product reduction with include_self
    input_dict = {
        "input": np.array([1, 2, 3, 4, 5], dtype=np.float32),
        "dim": 0,
        "index": np.array([0, 1, 2, 0, 3], dtype=np.int64),
        "src": np.array([1, 2, 3, 4, 5], dtype=np.float32),
        "reduce": "prod",
        "include_self": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: Mean reduction
    input_dict = {
        "input": np.zeros((4,), dtype=np.float32),
        "dim": 0,
        "index": np.array([0, 1, 3, 0], dtype=np.int64),
        "src": np.array([1, 2, 3, 4], dtype=np.float32),
        "reduce": "mean",
        "include_self": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4:  Min reduction with different data type and 2D input
    input_dict = {
        "input": np.array([[5, 6], [7, 8], [9, 10]], dtype=np.int32),
        "dim": 0,
        "index": np.array([[0], [1], [0]], dtype=np.int64),
        "src": np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32),
        "reduce": "amin",
        "include_self": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5: Max reduction with 3D input
    input_dict = {
        "input": np.zeros((2, 2, 2), dtype=np.float64),
        "dim": 1,
        "index": np.array([[[0, 0], [1, 1]], [[1, 1], [0, 0]]], dtype=np.int64),
        "src": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64),
        "reduce": "amax",
        "include_self": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.scatter_reduce_2"] = scatter_reduce_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.scatter_reduce_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.scatter_reduce_2'.")

check_valid('torch.scatter_reduce', generated_inputs['torch.scatter_reduce_2'], lib="torch")
