
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def cholesky_inputs():
    list_of_inputs = []

    # Input 1: Basic symmetric positive-definite matrix
    a = np.random.rand(3, 3)
    a = a @ a.T + np.eye(3) * 1e-3
    input_dict = {"input": a, "upper": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Batched input (multiple matrices)
    a = np.random.rand(2, 2, 2)
    a = a @ np.transpose(a, (0, 2, 1)) + np.eye(2) * 1e-3
    input_dict = {"input": a, "upper": True, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Larger matrix
    a = np.random.rand(5, 5)
    a = a @ a.T + np.eye(5) * 1e-3
    input_dict = {"input": a, "upper": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Batched input with different dimensions
    a = np.random.rand(4, 3, 3)
    a = a @ np.transpose(a, (0, 2, 1)) + np.eye(3) * 1e-3
    input_dict = {"input": a, "upper": True, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Single element matrix
    a = np.array([[1.0]])
    input_dict = {"input": a, "upper": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Another batched input with different dimensions
    a = np.random.rand(2, 4, 4)
    a = a @ np.transpose(a, (0, 2, 1)) + np.eye(4) * 1e-3
    input_dict = {"input": a, "upper": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.cholesky"] = cholesky_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.cholesky' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cholesky'.")

check_valid('torch.cholesky', generated_inputs['torch.cholesky'], lib="torch")
