
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def scatter_reduce_inputs():
    list_of_inputs = []

    # Input 1: Basic example with sum reduction
    input_dict = {
        "input": np.zeros((5,), dtype=np.float32),
        "dim": 0,
        "index": np.array([0, 1, 2, 0, 3]),
        "src": np.array([1, 2, 3, 4, 5], dtype=np.float32),
        "reduce": "sum",
        "include_self": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: sum reduction with include_self = True
    input_dict = {
        "input": np.array([1, 2, 3, 4, 5], dtype=np.float32),
        "dim": 0,
        "index": np.array([0, 1, 2, 0, 3]),
        "src": np.array([6, 7, 8, 9, 10], dtype=np.float32),
        "reduce": "sum",
        "include_self": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Two dimensional input with amin reduction
    input_dict = {
        "input": np.full((3, 3), 100.0, dtype=np.float32),
        "dim": 1,
        "index": np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]),
        "src": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        "reduce": "amin",
        "include_self": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Two dimensional input with amax reduction
    input_dict = {
        "input": np.zeros((3, 3), dtype=np.float32),
        "dim": 0,
        "index": np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
        "src": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        "reduce": "amax",
        "include_self": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Two dimensional input with mean reduction and include_self
    input_dict = {
        "input": np.ones((3, 3), dtype=np.float32) * 5,
        "dim": 1,
        "index": np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
        "src": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        "reduce": "mean",
        "include_self": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: product reduction with include_self = True
    input_dict = {
        "input": np.array([1, 2, 3, 4, 5], dtype=np.float32),
        "dim": 0,
        "index": np.array([0, 1, 2, 0, 3]),
        "src": np.array([6, 7, 8, 9, 10], dtype=np.float32),
        "reduce": "prod",
        "include_self": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Float64, sum reduction
    input_dict = {
        "input": np.zeros((5,), dtype=np.float64),
        "dim": 0,
        "index": np.array([0, 1, 2, 0, 3]),
        "src": np.array([1, 2, 3, 4, 5], dtype=np.float64),
        "reduce": "sum",
        "include_self": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.scatter_reduce_1"] = scatter_reduce_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.scatter_reduce_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.scatter_reduce_1'.")

check_valid('torch.scatter_reduce', generated_inputs['torch.scatter_reduce_1'], lib="torch")
