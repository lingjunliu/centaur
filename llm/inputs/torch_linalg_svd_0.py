
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def linalg_svd_inputs():
    list_of_inputs = []

    # Input 1: Basic tall matrix, reduced SVD
    input_dict_1 = {
        'A': np.random.rand(5, 3).astype(np.float32),
        'full_matrices': False,
        'driver': 'gesvdj',
        'out': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic tall matrix, full SVD
    input_dict_2 = {
        'A': np.random.rand(5, 3).astype(np.float64),
        'full_matrices': True,
        'driver': 'gesvd',
        'out': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Wide matrix, reduced SVD
    input_dict_3 = {
        'A': np.random.rand(3, 5).astype(np.float32),
        'full_matrices': False,
        'driver': 'gesvdj',
        'out': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Wide matrix, full SVD
    input_dict_4 = {
        'A': np.random.rand(3, 5).astype(np.float64),
        'full_matrices': True,
        'driver': 'gesvd',
        'out': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Square matrix, full SVD
    input_dict_5 = {
        'A': np.random.rand(4, 4).astype(np.float32),
        'full_matrices': True,
        'driver': 'gesvdj',
        'out': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Batched input, tall matrix, reduced SVD
    input_dict_6 = {
        'A': np.random.rand(2, 5, 3).astype(np.float32),
        'full_matrices': False,
        'driver': 'gesvda',
        'out': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Batched input, wide matrix, full SVD
    input_dict_7 = {
        'A': np.random.rand(3, 3, 5).astype(np.float64),
        'full_matrices': True,
        'driver': 'gesvd',
        'out': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Complex input (cfloat), reduced SVD
    input_dict_8 = {
        'A': (np.random.rand(4, 6) + 1j * np.random.rand(4, 6)).astype(np.complex64),
        'full_matrices': False,
        'driver': 'gesvdj',
        'out': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Complex input (cdouble), full SVD
    input_dict_9 = {
        'A': (np.random.rand(5, 5) + 1j * np.random.rand(5, 5)).astype(np.complex128),
        'full_matrices': True,
        'driver': 'gesvd',
        'out': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: Square matrix with negative values
    input_dict_10 = {
        'A': np.random.randn(6, 6).astype(np.float32),
        'full_matrices': False,
        'driver': 'gesvdj',
        'out': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))
    
    # Input 11: Using None for driver
    input_dict_11 = {
        'A': np.random.rand(5, 4).astype(np.float32),
        'full_matrices': True,
        'driver': None,
        'out': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    return list_of_inputs

generated_inputs["torch.linalg.svd"] = linalg_svd_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.linalg.svd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.svd'.")

check_valid('torch.linalg.svd', generated_inputs['torch.linalg.svd'], lib="torch", suffix=0)
