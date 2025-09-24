
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def matmul_inputs():
    list_of_inputs = []

    # Case 1: vector x vector
    tensor1 = torch.randn(3).numpy()
    tensor2 = torch.randn(3).numpy()
    input_dict = {"input": tensor1, "other": tensor2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: matrix x vector
    tensor1 = torch.randn(3, 4).numpy()
    tensor2 = torch.randn(4).numpy()
    input_dict = {"input": tensor1, "other": tensor2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: batched matrix x broadcasted vector
    tensor1 = torch.randn(10, 3, 4).numpy()
    tensor2 = torch.randn(4).numpy()
    input_dict = {"input": tensor1, "other": tensor2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: batched matrix x batched matrix
    tensor1 = torch.randn(10, 3, 4).numpy()
    tensor2 = torch.randn(10, 4, 5).numpy()
    input_dict = {"input": tensor1, "other": tensor2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: batched matrix x broadcasted matrix
    tensor1 = torch.randn(10, 3, 4).numpy()
    tensor2 = torch.randn(4, 5).numpy()
    input_dict = {"input": tensor1, "other": tensor2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.matmul"] = matmul_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.matmul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.matmul'.")

check_valid('torch.matmul', generated_inputs['torch.matmul'], lib="torch")
