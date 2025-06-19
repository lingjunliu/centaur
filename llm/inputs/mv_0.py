
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_mv_inputs():
    list_of_inputs = []

    # Example 1: Basic float tensors
    mat = np.random.randn(2, 3).astype(np.float32)
    vec = np.random.randn(3).astype(np.float32)
    input_dict = {"input": mat, "vec": vec, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Different dimensions, int tensors
    mat = np.random.randint(1, 5, size=(4, 2)).astype(np.int32)
    vec = np.random.randint(1, 5, size=(2)).astype(np.int32)
    input_dict = {"input": mat, "vec": vec, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: Negative values, double tensors
    mat = np.random.randn(3, 4).astype(np.float64) * -1
    vec = np.random.randn(4).astype(np.float64) * -1
    input_dict = {"input": mat, "vec": vec, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 4: Different shapes, float16
    mat = np.random.randn(5, 6).astype(np.float16)
    vec = np.random.randn(6).astype(np.float16)
    input_dict = {"input": mat, "vec": vec, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5: Larger dimensions
    mat = np.random.randn(10, 5).astype(np.float32)
    vec = np.random.randn(5).astype(np.float32)
    input_dict = {"input": mat, "vec": vec, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 6: Zero values
    mat = np.zeros((3, 2)).astype(np.float32)
    vec = np.zeros(2).astype(np.float32)
    input_dict = {"input": mat, "vec": vec, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.mv"] = torch_mv_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.mv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.mv'.")

check_valid('torch.mv', generated_inputs['torch.mv'], lib="torch")
