
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_linalg_eigvalsh_inputs():
    list_of_inputs = []

    # Input 1: Basic real symmetric matrix
    A = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    UPLO = 'L'
    out = np.zeros(2, dtype=np.float64)
    input_dict = {"A": A, "UPLO": UPLO, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Complex Hermitian matrix
    A = np.array([[2 + 0j, 1 - 1j], [1 + 1j, 3 + 0j]], dtype=np.complex128)
    UPLO = 'U'
    out = np.zeros(2, dtype=np.float64)
    input_dict = {"A": A, "UPLO": UPLO, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Batch of real symmetric matrices
    A = np.array([[[2.0, 1.0], [1.0, 3.0]], [[4.0, 2.0], [2.0, 5.0]]], dtype=np.float64)
    UPLO = 'L'
    out = np.zeros((2, 2), dtype=np.float64)
    input_dict = {"A": A, "UPLO": UPLO, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Larger real symmetric matrix
    A = np.array([[4.0, 1.0, 2.0], [1.0, 3.0, 0.0], [2.0, 0.0, 5.0]], dtype=np.float64)
    UPLO = 'U'
    out = np.zeros(3, dtype=np.float64)
    input_dict = {"A": A, "UPLO": UPLO, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Matrix with negative values
    A = np.array([[-2.0, 1.0], [1.0, -3.0]], dtype=np.float64)
    UPLO = 'L'
    out = np.zeros(2, dtype=np.float64)
    input_dict = {"A": A, "UPLO": UPLO, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 3D Batch of complex matrices
    A = np.array([[[2 + 0j, 1 - 1j], [1 + 1j, 3 + 0j]], [[4 + 0j, 2 - 2j], [2 + 2j, 5 + 0j]]], dtype=np.complex128)
    UPLO = 'L'
    out = np.zeros((2, 2), dtype=np.float64)
    input_dict = {"A": A, "UPLO": UPLO, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Single element matrix
    A = np.array([[5.0]], dtype=np.float64)
    UPLO = 'L'
    out = np.zeros(1, dtype=np.float64)
    input_dict = {"A": A, "UPLO": UPLO, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Matrix with zero values
    A = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    UPLO = 'U'
    out = np.zeros(2, dtype=np.float64)
    input_dict = {"A": A, "UPLO": UPLO, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.linalg.eigvalsh"] = torch_linalg_eigvalsh_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.eigvalsh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.eigvalsh'.")

check_valid('torch.linalg.eigvalsh', generated_inputs['torch.linalg.eigvalsh'], lib="torch", suffix=0)
