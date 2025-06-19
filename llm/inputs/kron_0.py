
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_kron_inputs():
    list_of_inputs = []

    # Example 1: Basic 2x2 matrices
    mat1 = np.eye(2)
    mat2 = np.ones((2, 2))
    input_dict = {"input": mat1, "other": mat2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Different values in matrices
    mat1 = np.eye(2)
    mat2 = np.arange(1, 5).reshape(2, 2)
    input_dict = {"input": mat1, "other": mat2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: Different sized matrices
    mat1 = np.array([[1, 2], [3, 4]])
    mat2 = np.array([[5, 6, 7], [8, 9, 10], [11, 12, 13]])
    input_dict = {"input": mat1, "other": mat2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: One dimension tensors
    vec1 = np.array([1, 2, 3])
    vec2 = np.array([4, 5])
    input_dict = {"input": vec1, "other": vec2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5: Three dimension tensors
    tensor1 = np.arange(1, 9).reshape(2, 2, 2)
    tensor2 = np.array([1, 2])
    input_dict = {"input": tensor1, "other": tensor2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.kron"] = torch_kron_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.kron' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.kron'.")

check_valid('torch.kron', generated_inputs['torch.kron'], lib="torch")
