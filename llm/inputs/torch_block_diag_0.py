
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def block_diag_inputs():
    list_of_inputs = []

    # Input 1: Basic test with two matrices
    tensors = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Three matrices of different sizes
    tensors = [np.array([[1]]), np.array([[2, 3], [4, 5]]), np.array([[6, 7, 8]])]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: One matrix
    tensors = [np.array([[1, 2, 3], [4, 5, 6]])]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Matrices with different data types (float)
    tensors = [np.array([[1.1, 2.2], [3.3, 4.4]]), np.array([[5.5, 6.6], [7.7, 8.8]])]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Matrices with different shapes and types (int and float)
    tensors = [np.array([[1, 2], [3, 4]]), np.array([[5.5]])]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Matrices with negative values
    tensors = [np.array([[-1, -2], [-3, -4]]), np.array([[-5, -6], [-7, -8]])]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.block_diag"] = block_diag_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.block_diag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.block_diag'.")

check_valid('torch.block_diag', generated_inputs['torch.block_diag'], lib="torch", suffix=0)
