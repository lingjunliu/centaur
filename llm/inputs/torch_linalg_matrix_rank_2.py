
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def matrix_rank_inputs():
    list_of_inputs = []

    # Input 1: Basic matrix
    input1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    tol1 = np.array([1e-8], dtype=np.float32)
    hermitian1 = False
    input_dict1 = {"input": input1, "tol": tol1, "hermitian": hermitian1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Singular matrix
    input2 = np.array([[1, 2], [2, 4]], dtype=np.float64)
    tol2 = np.array([1e-6], dtype=np.float64)
    hermitian2 = False
    input_dict2 = {"input": input2, "tol": tol2, "hermitian": hermitian2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Tall matrix
    input3 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    tol3 = np.array([1e-8], dtype=np.float32)
    hermitian3 = False
    input_dict3 = {"input": input3, "tol": tol3, "hermitian": hermitian3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Wide matrix
    input4 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    tol4 = np.array([1e-6], dtype=np.float64)
    hermitian4 = False
    input_dict4 = {"input": input4, "tol": tol4, "hermitian": hermitian4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Zero matrix
    input5 = np.array([[0, 0], [0, 0]], dtype=np.float32)
    tol5 = np.array([1e-8], dtype=np.float32)
    hermitian5 = False
    input_dict5 = {"input": input5, "tol": tol5, "hermitian": hermitian5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Identity matrix
    input6 = np.eye(3, dtype=np.float64)
    tol6 = np.array([1e-6], dtype=np.float64)
    hermitian6 = False
    input_dict6 = {"input": input6, "tol": tol6, "hermitian": hermitian6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

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
