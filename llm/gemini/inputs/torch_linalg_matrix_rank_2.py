
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def linalg_matrix_rank_inputs():
    list_of_inputs = []

    # The recurring errors indicate a discrepancy in the expected parameter name ('tol' vs 'rtol').
    # The parameter was renamed to 'rtol' in newer PyTorch versions.
    # The last error was `KeyError: 'rtol'`, so we will provide 'rtol'.
    # Assumed signature for the execution environment: {'input': 'tensor', 'rtol': 'tensor', 'hermitian': 'boolean'}

    # Input 1: Basic 2x2 full rank matrix, float32
    input_dict_1 = {
        'input': np.array([[1., 2.], [3., 4.]], dtype=np.float32),
        'rtol': np.array(0.0, dtype=np.float32),
        'hermitian': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2x3 rank deficient matrix, float64
    input_dict_2 = {
        'input': np.array([[1., 2., 3.], [2., 4., 6.]], dtype=np.float64),
        'rtol': np.array(0.0, dtype=np.float64),
        'hermitian': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 3x3 identity matrix (full rank, hermitian)
    input_dict_3 = {
        'input': np.eye(3, dtype=np.float32),
        'rtol': np.array(0.0, dtype=np.float32),
        'hermitian': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 3x3 zero matrix (rank 0), hermitian
    input_dict_4 = {
        'input': np.zeros((3, 3), dtype=np.float32),
        'rtol': np.array(0.0, dtype=np.float32),
        'hermitian': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Complex valued matrix, complex64
    input_dict_5 = {
        'input': np.array([[1.+1.j, 2.+2.j], [3.+3.j, 5.+5.j]], dtype=np.complex64),
        'rtol': np.array(0.0, dtype=np.float32),
        'hermitian': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    
    # Input 6: Batched input (2, 2, 3)
    input_dict_6 = {
        'input': np.array([[[1., 2., 0], [2., 4., 0]], 
                           [[1., 0., 0.], [0., 1., 0.]]], dtype=np.float64),
        'rtol': np.array(0.0, dtype=np.float64),
        'hermitian': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: Using `rtol` to change rank of a nearly singular matrix
    input_dict_7 = {
        'input': np.array([[1., 1.], [1., 1. + 1e-7]], dtype=np.float32),
        'rtol': np.array(1e-6, dtype=np.float32),
        'hermitian': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Using a smaller `rtol` which should not change the rank
    input_dict_8 = {
        'input': np.array([[1., 1.], [1., 1. + 1e-7]], dtype=np.float64),
        'rtol': np.array(1e-8, dtype=np.float64),
        'hermitian': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Hermitian complex matrix, complex128
    input_dict_9 = {
        'input': np.array([[2., 1.+1.j], [1.-1.j, 1.]], dtype=np.complex128),
        'rtol': np.array(0.0, dtype=np.float64),
        'hermitian': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: Tall matrix (4x2), full rank
    input_dict_10 = {
        'input': np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32),
        'rtol': np.array(0.0, dtype=np.float32),
        'hermitian': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["torch.linalg.matrix_rank_2"] = linalg_matrix_rank_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.linalg.matrix_rank_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.matrix_rank_2'.")

check_valid('torch.linalg.matrix_rank', generated_inputs['torch.linalg.matrix_rank_2'], lib="torch", suffix=2)
