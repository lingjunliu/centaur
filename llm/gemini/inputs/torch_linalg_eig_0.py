
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def linalg_eig_inputs():
    list_of_inputs = []

    # Input 1: Basic 2x2 real matrix (float32)
    A1 = np.array([[1.0, -1.0], [1.0, 1.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({'A': A1}))

    # Input 2: Basic 3x3 real matrix (float64) with negative values
    A2 = np.array([[0., 1., -2.], [-1., 0., 3.], [2., -3., 0.]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({'A': A2}))

    # Input 3: Basic 2x2 complex matrix (complex64)
    A3 = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({'A': A3}))

    # Input 4: Basic 3x3 complex matrix (complex128)
    A4 = (np.random.rand(3, 3) + 1j * np.random.rand(3, 3)).astype(np.complex128)
    list_of_inputs.append(copy.deepcopy({'A': A4}))

    # Input 5: Batched real matrices (float32)
    A5 = np.random.randn(2, 3, 3).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({'A': A5}))

    # Input 6: Batched complex matrices (complex128)
    A6 = (np.random.randn(3, 2, 2) + 1j * np.random.randn(3, 2, 2)).astype(np.complex128)
    list_of_inputs.append(copy.deepcopy({'A': A6}))

    # Input 7: Identity matrix (float64), has repeated eigenvalues
    A7 = np.eye(4, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({'A': A7}))

    # Input 8: Zero matrix (float32)
    A8 = np.zeros((2, 2), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({'A': A8}))
    
    # Input 9: A larger 5x5 real matrix (float64)
    A9 = np.random.randn(5, 5).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({'A': A9}))
    
    # Input 10: A multi-dimensional batch (2, 1, 3, 3) of complex matrices (complex64)
    A10 = (np.random.randn(2, 1, 3, 3) + 1j * np.random.randn(2, 1, 3, 3)).astype(np.complex64)
    list_of_inputs.append(copy.deepcopy({'A': A10}))

    return list_of_inputs

generated_inputs["torch.linalg.eig"] = linalg_eig_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.linalg.eig' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.eig'.")

check_valid('torch.linalg.eig', generated_inputs['torch.linalg.eig'], lib="torch", suffix=0)
