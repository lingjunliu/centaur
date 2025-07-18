
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def linalg_matrix_norm_inputs():
    list_of_inputs = []

    # Input 1: Basic spectral norm (ord=2), float32
    input_dict = {
        'input': np.arange(1, 10, dtype=np.float32).reshape(3, 3),
        'ord': [2],
        'dim': (0, 1),
        'keepdim': False,
        'out': np.array(0.0, dtype=np.float32),
        'dtype': np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Batched 1-norm, float64, with keepdim=True
    input_dict = {
        'input': np.random.rand(2, 4, 5).astype(np.float64),
        'ord': [1],
        'dim': (1, 2),
        'keepdim': True,
        'out': np.empty((2, 1, 1), dtype=np.float64),
        'dtype': np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: -1 norm on a matrix with negative values
    input_dict = {
        'input': (np.random.rand(4, 4) - 0.5).astype(np.float32),
        'ord': [-1],
        'dim': (0, 1),
        'keepdim': False,
        'out': np.array(0.0, dtype=np.float32),
        'dtype': np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Spectral norm (2) on a complex batch of matrices
    input_dict = {
        'input': (np.random.randn(3, 5, 2) + 1j * np.random.randn(3, 5, 2)).astype(np.complex64),
        'ord': [2],
        'dim': (-2, -1),
        'keepdim': False,
        'out': np.empty((3,), dtype=np.float32),
        'dtype': np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: -2 norm (smallest singular value) on complex data
    input_dict = {
        'input': (np.random.rand(4, 3) + 1j*np.random.rand(4,3)).astype(np.complex128),
        'ord': [-2],
        'dim': (0, 1),
        'keepdim': False,
        'out': np.array(0.0, dtype=np.float64),
        'dtype': np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Infinity norm on 4D tensor with keepdim=True
    input_dict = {
        'input': np.arange(1, 25, dtype=np.float64).reshape(2, 2, 2, 3),
        'ord': [float('inf')],
        'dim': (2, 3),
        'keepdim': True,
        'out': np.empty((2, 2, 1, 1), dtype=np.float64),
        'dtype': np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: -Infinity norm (min row sum)
    input_dict = {
        'input': (np.random.rand(3, 5) * 10 + 1).astype(np.float32),
        'ord': [float('-inf')],
        'dim': (0, 1),
        'keepdim': False,
        'out': np.array(0.0, dtype=np.float32),
        'dtype': np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Using the 'out' parameter with a correctly sized tensor for -1 norm
    input_data_out = np.random.rand(2, 5, 4).astype(np.float64)
    out_tensor = np.empty((2,), dtype=np.float64)
    input_dict = {
        'input': input_data_out,
        'ord': [-1],
        'dim': (1, 2),
        'keepdim': False,
        'out': out_tensor,
        'dtype': np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: dtype promotion from float32 to float64
    input_dict = {
        'input': np.random.rand(4, 4).astype(np.float32),
        'ord': [1],
        'dim': (0, 1),
        'keepdim': False,
        'out': np.array(0.0, dtype=np.float64),
        'dtype': np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty dimension in input tensor
    input_dict = {
        'input': np.zeros((3, 0, 5)).astype(np.float32),
        'ord': [2],
        'dim': (1, 2),
        'keepdim': False,
        'out': np.empty((3,), dtype=np.float32),
        'dtype': np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.linalg.matrix_norm_3"] = linalg_matrix_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.linalg.matrix_norm_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.matrix_norm_3'.")

check_valid('torch.linalg.matrix_norm', generated_inputs['torch.linalg.matrix_norm_3'], lib="torch", suffix=3)
