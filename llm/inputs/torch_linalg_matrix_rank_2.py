
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def matrix_rank_inputs():
    list_of_inputs = []

    # Input 1: Simple matrix with full rank
    input = np.array([[1.0, 2.0], [3.0, 4.0]])
    tol = np.array([1e-8])
    hermitian = False
    input_dict = {"input": input, "tol": tol, "hermitian": hermitian}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Singular matrix
    input = np.array([[1.0, 2.0], [2.0, 4.0]])
    tol = np.array([1e-8])
    hermitian = False
    input_dict = {"input": input, "tol": tol, "hermitian": hermitian}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Zero matrix
    input = np.array([[0.0, 0.0], [0.0, 0.0]])
    tol = np.array([1e-8])
    hermitian = False
    input_dict = {"input": input, "tol": tol, "hermitian": hermitian}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tall matrix with full column rank
    input = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    tol = np.array([1e-8])
    hermitian = False
    input_dict = {"input": input, "tol": tol, "hermitian": hermitian}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Wide matrix with full row rank
    input = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    tol = np.array([1e-8])
    hermitian = False
    input_dict = {"input": input, "tol": tol, "hermitian": hermitian}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Matrix with small tolerance
    input = np.array([[1.0, 0.0], [0.0, 1e-9]])
    tol = np.array([1e-6])
    hermitian = False
    input_dict = {"input": input, "tol": tol, "hermitian": hermitian}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Hermitian matrix
    input = np.array([[1.0, 2.0j], [-2.0j, 1.0]])
    tol = np.array([1e-8])
    hermitian = True
    input_dict = {"input": input, "tol": tol, "hermitian": hermitian}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Higher dimensional array (3D)
    input = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    tol = np.array([1e-8])
    hermitian = False
    input_dict = {"input": input, "tol": tol, "hermitian": hermitian}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Rectangular matrix close to singular
    input = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0 + 1e-7]])
    tol = np.array([1e-5])
    hermitian = False
    input_dict = {"input": input, "tol": tol, "hermitian": hermitian}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.linalg.matrix_rank_2"] = matrix_rank_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.matrix_rank_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.matrix_rank_2'.")

check_valid('torch.linalg.matrix_rank', generated_inputs['torch.linalg.matrix_rank_2'], lib="torch", suffix=2)
