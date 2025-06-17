
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def bmm_inputs():
    list_of_inputs = []

    # Example 1: Basic float tensors
    input1 = np.random.randn(10, 3, 4).astype(np.float32)
    mat2_1 = np.random.randn(10, 4, 5).astype(np.float32)
    input_dict1 = {"input": input1, "mat2": mat2_1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Example 2: Different batch size and dimensions
    input2 = np.random.randn(5, 2, 3).astype(np.float32)
    mat2_2 = np.random.randn(5, 3, 2).astype(np.float32)
    input_dict2 = {"input": input2, "mat2": mat2_2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Example 3: Integer tensors
    input3 = np.random.randint(-5, 5, size=(3, 4, 2)).astype(np.int32)
    mat2_3 = np.random.randint(-5, 5, size=(3, 2, 3)).astype(np.int32)
    input_dict3 = {"input": input3, "mat2": mat2_3, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Example 4: Negative values
    input4 = np.random.randn(2, 5, 5).astype(np.float32) * -1
    mat2_4 = np.random.randn(2, 5, 5).astype(np.float32) * -1
    input_dict4 = {"input": input4, "mat2": mat2_4, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Example 5: Larger dimensions
    input5 = np.random.randn(4, 10, 8).astype(np.float32)
    mat2_5 = np.random.randn(4, 8, 12).astype(np.float32)
    input_dict5 = {"input": input5, "mat2": mat2_5, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.bmm"] = bmm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.bmm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.bmm'.")

check_valid('torch.bmm', generated_inputs['torch.bmm'], lib="torch")
