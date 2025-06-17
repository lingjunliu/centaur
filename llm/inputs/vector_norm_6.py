
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def vector_norm_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float tensor
    input1 = np.array([1.0, 2.0, 3.0])
    input_dict1 = {
        "input": input1,
        "ord": 2.0,
        "dim": None,
        "keepdim": False,
        "dtype": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Integer tensor with specified dimension, cast to float
    input2 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    input_dict2 = {
        "input": input2,
        "ord": 1.0,
        "dim": [0],
        "keepdim": True,
        "dtype": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Negative values and different norm
    input3 = np.array([-1.0, 2.0, -3.0])
    input_dict3 = {
        "input": input3,
        "ord": np.inf,
        "dim": None,
        "keepdim": False,
        "dtype": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Multi-dimensional tensor
    input4 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    input_dict4 = {
        "input": input4,
        "ord": 2.0,
        "dim": [1, 2],
        "keepdim": False,
        "dtype": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Specify the dtype
    input5 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict5 = {
        "input": input5,
        "ord": 2.0,
        "dim": None,
        "keepdim": False,
        "dtype": torch.float64,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Use negative ord
    input6 = np.array([1.0, 2.0, 3.0])
    input_dict6 = {
        "input": input6,
        "ord": -2.0,
        "dim": None,
        "keepdim": False,
        "dtype": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Complex tensor
    input7 = np.array([1 + 1j, 2 - 2j, 3 + 0j])
    input_dict7 = {
        "input": input7,
        "ord": 2.0,
        "dim": None,
        "keepdim": False,
        "dtype": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.linalg.vector_norm_6"] = vector_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.vector_norm_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.vector_norm_6'.")

check_valid('torch.linalg.vector_norm', generated_inputs['torch.linalg.vector_norm_6'], lib="torch")
