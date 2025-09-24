
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_einsum_inputs():
    list_of_inputs = []

    # Example 1: Matrix multiplication
    A = np.random.randn(2, 3).astype(np.float32)
    B = np.random.randn(3, 4).astype(np.float32)
    input_dict = {
        "equation": "ij,jk->ik",
        "*operands": [A, B]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Trace of a matrix
    A = np.random.randn(4, 4).astype(np.float32)
    input_dict = {
        "equation": "ii",
        "*operands": [A]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: Batch matrix multiplication
    A = np.random.randn(3, 2, 5).astype(np.float32)
    B = np.random.randn(3, 5, 4).astype(np.float32)
    input_dict = {
        "equation": "bij,bjk->bik",
        "*operands": [A, B]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: Outer product
    x = np.random.randn(5).astype(np.float32)
    y = np.random.randn(4).astype(np.float32)
    input_dict = {
        "equation": "i,j->ij",
        "*operands": [x, y]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 5: Bilinear operation
    A = np.random.randn(3, 5, 4).astype(np.float32)
    l = np.random.randn(2, 5).astype(np.float32)
    r = np.random.randn(2, 4).astype(np.float32)
    input_dict = {
        "equation": "bn,anm,bm->ba",
        "*operands": [l, A, r]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.einsum_1"] = torch_einsum_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.einsum_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.einsum_1'.")

check_valid('torch.einsum', generated_inputs['torch.einsum_1'], lib="torch")
