
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_mm_inputs():
    list_of_inputs = []

    # Test case 1: Basic float matrices
    mat1 = np.random.randn(2, 3).astype(np.float32)
    mat2 = np.random.randn(3, 4).astype(np.float32)
    input_dict = {"input": mat1, "mat2": mat2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: Integer matrices
    mat1 = np.random.randint(-5, 5, size=(5, 2)).astype(np.int32)
    mat2 = np.random.randint(-5, 5, size=(2, 6)).astype(np.int32)
    input_dict = {"input": mat1, "mat2": mat2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: Negative values and larger dimensions
    mat1 = np.random.randn(4, 5).astype(np.float64) * -1
    mat2 = np.random.randn(5, 3).astype(np.float64) * -1
    input_dict = {"input": mat1, "mat2": mat2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: Output tensor provided
    mat1 = np.random.randn(3, 3).astype(np.float32)
    mat2 = np.random.randn(3, 2).astype(np.float32)
    out = np.zeros((3, 2)).astype(np.float32)
    input_dict = {"input": mat1, "mat2": mat2, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 5: Different shaped matrices
    mat1 = np.random.randn(1, 5).astype(np.float32)
    mat2 = np.random.randn(5, 1).astype(np.float32)
    input_dict = {"input": mat1, "mat2": mat2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.mm"] = torch_mm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.mm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.mm'.")

check_valid('torch.mm', generated_inputs['torch.mm'], lib="torch")
