
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def matrix_power_inputs():
    list_of_inputs = []

    # Input 1: 2x2 matrix, n = 0
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    n = 0
    input_dict = {"A": A, "n": n}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3x3 matrix, n = 1
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    n = 1
    input_dict = {"A": A, "n": n}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2x2 matrix, n = 2
    A = np.array([[0.0, 1.0], [1.0, 0.0]])
    n = 2
    input_dict = {"A": A, "n": n}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4x4 matrix, n = -1 (invertible)
    A = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 4.0]])
    n = -1
    input_dict = {"A": A, "n": n}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2x2 complex matrix, n = 2
    A = np.array([[1 + 1j, 2 - 1j], [3 + 0j, 4 - 2j]])
    n = 2
    input_dict = {"A": A, "n": n}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3x3 matrix, n = 3
    A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    n = 3
    input_dict = {"A": A, "n": n}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 2x2 matrix, n = -2 (invertible)
    A = np.array([[2.0, 1.0], [1.0, 1.0]])
    n = -2
    input_dict = {"A": A, "n": n}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3x3 complex matrix, n = -1 (invertible)
    A = np.array([[1 + 0j, 0 + 0j, 0 + 0j], [0 + 0j, 2 + 0j, 0 + 0j], [0 + 0j, 0 + 0j, 3 + 0j]])
    n = -1
    input_dict = {"A": A, "n": n}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: small matrix, large n
    A = np.array([[1.0, 1.0], [0.0, 1.0]])
    n = 10
    input_dict = {"A": A, "n": n}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: identity matrix
    A = np.eye(3)
    n = 5
    input_dict = {"A": A, "n": n}
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.linalg.matrix_power"] = matrix_power_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.matrix_power' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.matrix_power'.")

check_valid('torch.linalg.matrix_power', generated_inputs['torch.linalg.matrix_power'], lib="torch", suffix=0)
