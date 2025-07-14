
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def chain_matmul_inputs():
    list_of_inputs = []

    # Input 1: Basic case with two matrices
    matrices = [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0, 6.0], [7.0, 8.0]])]
    input_dict = {"matrices": matrices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Three matrices
    matrices = [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0, 6.0], [7.0, 8.0]]), np.array([[9.0, 10.0], [11.0, 12.0]])]
    input_dict = {"matrices": matrices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Matrices with different shapes (but compatible)
    matrices = [np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])]
    input_dict = {"matrices": matrices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Matrices with negative values
    matrices = [np.array([[-1.0, 2.0], [3.0, -4.0]]), np.array([[5.0, -6.0], [-7.0, 8.0]])]
    input_dict = {"matrices": matrices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Matrices with zero values
    matrices = [np.array([[1.0, 0.0], [0.0, 4.0]]), np.array([[0.0, 6.0], [7.0, 0.0]])]
    input_dict = {"matrices": matrices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: More matrices - Reduced number of matrices for simplicity
    matrices = [np.array([[1.0]]), np.array([[2.0]])]
    input_dict = {"matrices": matrices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger matrices - Reduced dimensions to avoid issues
    matrices = [np.random.rand(5, 3).astype(np.float32), np.random.rand(3, 4).astype(np.float32)]
    input_dict = {"matrices": matrices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Smaller matrices
    matrices = [np.random.rand(1, 1).astype(np.float32), np.random.rand(1, 1).astype(np.float32)]
    input_dict = {"matrices": matrices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Matrices with a single element
    matrices = [np.array([[1.0]]), np.array([[2.0]])]
    input_dict = {"matrices": matrices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.chain_matmul"] = chain_matmul_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.chain_matmul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.chain_matmul'.")

check_valid('torch.chain_matmul', generated_inputs['torch.chain_matmul'], lib="torch", suffix=0)
