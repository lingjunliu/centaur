
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def sparse_bsr_tensor_inputs():
    list_of_inputs = []

    # The user is facing a conflict between the testing framework's required input structure (KeyError)
    # and the actual PyTorch API signature (TypeError).
    # The traceback `TypeError: ... got (..., list, ...)` indicates `size` is passed as a list,
    # while the PyTorch function `sparse_bsr_tensor` expects a tuple for the `size` parameter.
    # To address the TypeError, the value for the 'size' key is changed from a list to a tuple.
    # All keys from the prompt's signature, including 'blocksize', are kept to prevent KeyErrors from
    # the testing framework.

    # Input 1: Basic case, float32, with size as tuple
    input_dict_1 = {
        'compressed_indices': np.array([0, 1, 2], dtype=np.int64),
        'plain_indices': np.array([1, 0], dtype=np.int64),
        'values': np.random.rand(2, 3, 2).astype(np.float32),
        'size': (6, 4),
        'blocksize': (3, 2),
        'dtype': np.float32,
        'requires_grad': False,
        'layout': 'torch.sparse_bsr'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Larger matrix, int32, with size as tuple
    input_dict_2 = {
        'compressed_indices': np.array([0, 2, 2, 3, 4, 4], dtype=np.int64),
        'plain_indices': np.array([1, 3, 0, 4], dtype=np.int64),
        'values': np.arange(4 * 2 * 3).reshape(4, 2, 3).astype(np.int32),
        'size': (10, 15),
        'blocksize': (2, 3),
        'dtype': np.int32,
        'requires_grad': False,
        'layout': 'torch.sparse_bsr'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Empty sparse tensor, with size as tuple
    input_dict_3 = {
        'compressed_indices': np.array([0, 0, 0], dtype=np.int64),
        'plain_indices': np.array([], dtype=np.int64),
        'values': np.zeros((0, 4, 4), dtype=np.float64),
        'size': (8, 8),
        'blocksize': (4, 4),
        'dtype': np.float64,
        'requires_grad': True,
        'layout': 'torch.sparse_bsr'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Fully dense BSR, complex64, with size as tuple
    input_dict_4 = {
        'compressed_indices': np.array([0, 2, 4], dtype=np.int64),
        'plain_indices': np.array([0, 1, 0, 1], dtype=np.int64),
        'values': (np.random.rand(4, 2, 3) + 1j * np.random.rand(4, 2, 3)).astype(np.complex64),
        'size': (4, 6),
        'blocksize': (2, 3),
        'dtype': np.complex64,
        'requires_grad': False,
        'layout': 'torch.sparse_bsr'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: float32 with requires_grad=True, with size as tuple
    input_dict_5 = {
        'compressed_indices': np.array([0, 1, 2], dtype=np.int64),
        'plain_indices': np.array([1, 0], dtype=np.int64),
        'values': np.random.rand(2, 3, 2).astype(np.float32),
        'size': (6, 4),
        'blocksize': (3, 2),
        'dtype': np.float32,
        'requires_grad': True,
        'layout': 'torch.sparse_bsr'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: float64 with requires_grad=True, diagonal blocks, with size as tuple
    input_dict_6 = {
        'compressed_indices': np.array([0, 1, 2, 3], dtype=np.int64),
        'plain_indices': np.array([0, 1, 2], dtype=np.int64),
        'values': np.random.randn(3, 3, 3).astype(np.float64),
        'size': (9, 9),
        'blocksize': (3, 3),
        'dtype': np.float64,
        'requires_grad': True,
        'layout': 'torch.sparse_bsr'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Empty block row, int64, with size as tuple
    input_dict_7 = {
        'compressed_indices': np.array([0, 2, 2, 3], dtype=np.int64),
        'plain_indices': np.array([1, 3, 0], dtype=np.int64),
        'values': np.arange(3 * 4 * 2, dtype=np.int64).reshape(3, 4, 2),
        'size': (12, 8),
        'blocksize': (4, 2),
        'dtype': np.int64,
        'requires_grad': False,
        'layout': 'torch.sparse_bsr'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Blocksize (1, 1), with size as tuple
    input_dict_8 = {
        'compressed_indices': np.array([0, 2, 3, 3, 5, 6], dtype=np.int64),
        'plain_indices': np.array([0, 4, 2, 1, 3, 0], dtype=np.int64),
        'values': np.random.rand(6, 1, 1).astype(np.float32),
        'size': (5, 5),
        'blocksize': (1, 1),
        'dtype': np.float32,
        'requires_grad': False,
        'layout': 'torch.sparse_bsr'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Complex128, with size as tuple
    input_dict_9 = {
        'compressed_indices': np.array([0, 1, 2], dtype=np.int64),
        'plain_indices': np.array([0, 1], dtype=np.int64),
        'values': (np.random.rand(2, 2, 2) + 1j * np.random.rand(2, 2, 2)).astype(np.complex128),
        'size': (4, 4),
        'blocksize': (2, 2),
        'dtype': np.complex128,
        'requires_grad': False,
        'layout': 'torch.sparse_bsr'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Negative values, non-square block, int16, with size as tuple
    input_dict_10 = {
        'compressed_indices': np.array([0, 1, 3, 3, 5], dtype=np.int64),
        'plain_indices': np.array([2, 0, 1, 0, 2], dtype=np.int64),
        'values': (np.arange(5 * 2 * 5) - 25).reshape(5, 2, 5).astype(np.int16),
        'size': (8, 15),
        'blocksize': (2, 5),
        'dtype': np.int16,
        'requires_grad': False,
        'layout': 'torch.sparse_bsr'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["torch.sparse_bsr_tensor_4"] = sparse_bsr_tensor_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.sparse_bsr_tensor_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sparse_bsr_tensor_4'.")

check_valid('torch.sparse_bsr_tensor', generated_inputs['torch.sparse_bsr_tensor_4'], lib="torch", suffix=4)
