
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def matrix_exp_inputs():
    list_of_inputs = []

    # Input 1: Basic square matrix
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Negative values
    input2 = np.array([[-1.0, 0.5], [0.2, -2.0]], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Complex numbers
    input3 = np.array([[1 + 1j, 2 - 1j], [3 + 0j, 4 - 2j]], dtype=np.complex64)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Larger matrix
    input4 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Batch of matrices
    input5 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.matrix_exp"] = matrix_exp_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.matrix_exp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.matrix_exp'.")

check_valid('torch.matrix_exp', generated_inputs['torch.matrix_exp'], lib="torch")
