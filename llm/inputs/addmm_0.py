
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def addmm_inputs():
    list_of_inputs = []

    # Example 1: Basic case with float tensors
    input1 = np.random.randn(2, 3).astype(np.float32)
    mat11 = np.random.randn(2, 4).astype(np.float32)
    mat21 = np.random.randn(4, 3).astype(np.float32)
    input_dict1 = {
        "input": input1,
        "mat1": mat11,
        "mat2": mat21,
        "beta": 1.0,
        "alpha": 1.0,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Example 2: Integer tensors
    input2 = np.random.randint(1, 10, size=(3, 4)).astype(np.int32)
    mat12 = np.random.randint(1, 10, size=(3, 5)).astype(np.int32)
    mat22 = np.random.randint(1, 10, size=(5, 4)).astype(np.int32)
    input_dict2 = {
        "input": input2,
        "mat1": mat12,
        "mat2": mat22,
        "beta": 2,
        "alpha": 3,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Example 3: Negative values, beta = 0
    input3 = np.random.randn(4, 2).astype(np.float64)
    mat13 = np.random.randn(4, 5).astype(np.float64)
    mat23 = np.random.randn(5, 2).astype(np.float64)
    input_dict3 = {
        "input": input3,
        "mat1": mat13,
        "mat2": mat23,
        "beta": 0.0,
        "alpha": -1.0,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Example 4: Different alpha and beta values, providing an output tensor
    input4 = np.random.randn(5, 5).astype(np.float32)
    mat14 = np.random.randn(5, 3).astype(np.float32)
    mat24 = np.random.randn(3, 5).astype(np.float32)
    out4 = np.random.randn(5, 5).astype(np.float32)
    input_dict4 = {
        "input": input4,
        "mat1": mat14,
        "mat2": mat24,
        "beta": 0.5,
        "alpha": 1.5,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Example 5: Broadcasting input
    input5 = np.random.randn(1, 3).astype(np.float32)
    mat15 = np.random.randn(2, 4).astype(np.float32)
    mat25 = np.random.randn(4, 3).astype(np.float32)
    input_dict5 = {
        "input": input5,
        "mat1": mat15,
        "mat2": mat25,
        "beta": 1.0,
        "alpha": 1.0,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.addmm"] = addmm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.addmm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.addmm'.")

check_valid('torch.addmm', generated_inputs['torch.addmm'], lib="torch")
