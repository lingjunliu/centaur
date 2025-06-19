
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def logdet_inputs():
    list_of_inputs = []

    # Input 1: Basic square matrix
    A = np.random.rand(3, 3)
    input_dict = {"input": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Batch of square matrices
    A = np.random.rand(2, 4, 4)
    input_dict = {"input": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Matrix with negative determinant (should return NaN if converted to torch and logdet is calculated)
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    input_dict = {"input": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Matrix with zero determinant (should return -inf if converted to torch and logdet is calculated)
    A = np.array([[1.0, 1.0], [1.0, 1.0]])
    input_dict = {"input": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger batch of matrices
    A = np.random.rand(5, 5, 5)
    input_dict = {"input": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.logdet"] = logdet_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.logdet' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logdet'.")

check_valid('torch.logdet', generated_inputs['torch.logdet'], lib="torch")
