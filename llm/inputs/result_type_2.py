
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def result_type_inputs():
    list_of_inputs = []

    # Example 1: Basic float and int tensors
    tensor1 = np.array([1.0, 2.0], dtype=np.float32)
    tensor2 = np.array([3, 4], dtype=np.int32)
    input_dict = {"tensor1": tensor1, "tensor2": tensor2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Different float precisions
    tensor1 = np.array([1.0, 2.0], dtype=np.float64)
    tensor2 = np.array([3.0, 4.0], dtype=np.float16)
    input_dict = {"tensor1": tensor1, "tensor2": tensor2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: Integer and bool
    tensor1 = np.array([1, 2], dtype=np.int64)
    tensor2 = np.array([True, False], dtype=np.bool_)
    input_dict = {"tensor1": tensor1, "tensor2": tensor2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: Complex and Float
    tensor1 = np.array([1.0 + 1j, 2.0 + 2j], dtype=np.complex64)
    tensor2 = np.array([3.0, 4.0], dtype=np.float32)
    input_dict = {"tensor1": tensor1, "tensor2": tensor2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5: Higher dimension tensors
    tensor1 = np.random.rand(2, 3).astype(np.float32)
    tensor2 = np.random.randint(0, 10, size=(2, 3)).astype(np.int32)
    input_dict = {"tensor1": tensor1, "tensor2": tensor2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.result_type_2"] = result_type_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.result_type_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.result_type_2'.")

check_valid('torch.result_type', generated_inputs['torch.result_type_2'], lib="torch")
