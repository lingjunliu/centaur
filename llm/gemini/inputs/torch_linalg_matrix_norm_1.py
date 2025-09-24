
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def linalg_matrix_norm_inputs():
    list_of_inputs = []
    
    # All dictionaries must contain all 6 keys as per the provided signature.
    # The 'out' tensor must have the correct shape and dtype for the operation.
    # The 'dtype' parameter dictates the computation dtype and thus the output dtype.
    # The dtype values are changed to np.dtype('...') objects to fix the `dtype=type` error.

    # Input 1: Basic 2D float32, keepdim=False
    input_1 = np.array([[1., -2.], [3., -4.]], dtype=np.float32)
    out_1 = np.empty((), dtype=np.float32)
    list_of_inputs.append({
        'input': input_1,
        'ord': 2,
        'dim': (-2, -1),
        'keepdim': False,
        'out': out_1,
        'dtype': np.dtype('float32')
    })

    # Input 2: 3D tensor (batch of matrices), float64, keepdim=True
    input_2 = np.random.rand(2, 3, 4).astype(np.float64)
    out_2 = np.empty((2, 1, 1), dtype=np.float64)
    list_of_inputs.append({
        'input': input_2,
        'ord': 1,
        'dim': (1, 2),
        'keepdim': True,
        'out': out_2,
        'dtype': np.dtype('float64')
    })

    # Input 3: Complex64 input, results in float32 output
    input_3 = (np.random.rand(2, 5, 6) + 1j * np.random.rand(2, 5, 6)).astype(np.complex64)
    out_3 = np.empty((2,), dtype=np.float32)
    list_of_inputs.append({
        'input': input_3,
        'ord': -1,
        'dim': (-2, -1),
        'keepdim': False,
        'out': out_3,
        'dtype': np.dtype('float32')
    })

    # Input 4: Complex128 input, results in float64 output, keepdim=True
    input_4 = (np.random.rand(3, 4, 2) + 1j * np.random.rand(3, 4, 2)).astype(np.complex128)
    out_4 = np.empty((3, 1, 1), dtype=np.float64)
    list_of_inputs.append({
        'input': input_4,
        'ord': -2,
        'dim': (1, 2),
        'keepdim': True,
        'out': out_4,
        'dtype': np.dtype('float64')
    })

    # Input 5: Upcasting with `dtype` parameter (float32 -> float64)
    input_5 = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    out_5 = np.empty((), dtype=np.float64)
    list_of_inputs.append({
        'input': input_5,
        'ord': 2,
        'dim': (-2, -1),
        'keepdim': False,
        'out': out_5,
        'dtype': np.dtype('float64')
    })

    # Input 6: 4D tensor, non-default dim, keepdim=False
    input_6 = np.random.rand(2, 3, 4, 5).astype(np.float32)
    out_6 = np.empty((2, 4), dtype=np.float32)
    list_of_inputs.append({
        'input': input_6,
        'ord': 1,
        'dim': (1, 3),
        'keepdim': False,
        'out': out_6,
        'dtype': np.dtype('float32')
    })
    
    # Input 7: 4D tensor, non-default dim, keepdim=True
    input_7 = np.random.rand(2, 3, 4, 5).astype(np.float64)
    out_7 = np.empty((1, 3, 1, 5), dtype=np.float64)
    list_of_inputs.append({
        'input': input_7,
        'ord': -1,
        'dim': (0, 2),
        'keepdim': True,
        'out': out_7,
        'dtype': np.dtype('float64')
    })

    # Input 8: Upcasting complex computation (complex64 -> float64) via dtype
    input_8 = (np.random.rand(3, 4) + 1j * np.random.rand(3, 4)).astype(np.complex64)
    out_8 = np.empty((), dtype=np.float64)
    list_of_inputs.append({
        'input': input_8,
        'ord': 2,
        'dim': (0, 1),
        'keepdim': False,
        'out': out_8,
        'dtype': np.dtype('float64') # result dtype for complex input
    })

    # Input 9: High-dimensional case with negative dim indices
    input_9 = np.random.randn(2, 2, 2, 2).astype(np.float32)
    out_9 = np.empty((2, 2), dtype=np.float32)
    list_of_inputs.append({
        'input': input_9,
        'ord': -2,
        'dim': (-3, -1),
        'keepdim': False,
        'out': out_9,
        'dtype': np.dtype('float32')
    })

    # Input 10: Minimal 2D case, float64, keepdim=True
    input_10 = np.array([[3.]], dtype=np.float64)
    out_10 = np.empty((1, 1), dtype=np.float64)
    list_of_inputs.append({
        'input': input_10,
        'ord': 1,
        'dim': (0, 1),
        'keepdim': True,
        'out': out_10,
        'dtype': np.dtype('float64')
    })

    return list_of_inputs

generated_inputs["torch.linalg.matrix_norm_1"] = linalg_matrix_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.linalg.matrix_norm_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.matrix_norm_1'.")

check_valid('torch.linalg.matrix_norm', generated_inputs['torch.linalg.matrix_norm_1'], lib="torch", suffix=1)
