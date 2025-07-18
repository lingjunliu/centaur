
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def sparse_bsr_tensor_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float32
    input_dict_1 = {
        'compressed_indices': np.array([0, 1, 2], dtype=np.int64),
        'plain_indices': np.array([1, 2], dtype=np.int64),
        'values': np.arange(1, 9, dtype=np.float32).reshape(2, 2, 2),
        'size': [4, 6],
        'blocksize': (2, 2),
        'dtype': np.float32,
        'requires_grad': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Empty block row, float64
    input_dict_2 = {
        'compressed_indices': np.array([0, 1, 1, 2], dtype=np.int64),
        'plain_indices': np.array([0, 1], dtype=np.int64),
        'values': np.random.randn(2, 2, 2).astype(np.float64),
        'size': [6, 4],
        'blocksize': (2, 2),
        'dtype': np.float64,
        'requires_grad': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Fully dense BSR with int32
    input_dict_3 = {
        'compressed_indices': np.array([0, 2, 4], dtype=np.int64),
        'plain_indices': np.array([0, 1, 0, 1], dtype=np.int64),
        'values': np.arange(8, dtype=np.int32).reshape(4, 1, 2),
        'size': [2, 4],
        'blocksize': (1, 2),
        'dtype': np.int32,
        'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Fully sparse (all zeros)
    input_dict_4 = {
        'compressed_indices': np.array([0, 0, 0], dtype=np.int64),
        'plain_indices': np.array([], dtype=np.int64),
        'values': np.zeros((0, 4, 5), dtype=np.float32),
        'size': [8, 10],
        'blocksize': (4, 5),
        'dtype': np.float32,
        'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Large dimensions with complex numbers
    input_dict_5 = {
        'compressed_indices': np.array([0, 1, 1, 3, 3], dtype=np.int64),
        'plain_indices': np.array([2, 0, 3], dtype=np.int64),
        'values': (np.random.randn(3, 3, 4) + 1j * np.random.randn(3, 3, 4)).astype(np.complex64),
        'size': [12, 16],
        'blocksize': (3, 4),
        'dtype': np.complex64,
        'requires_grad': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Non-square blocks with negative values
    input_dict_6 = {
        'compressed_indices': np.array([0, 1, 2], dtype=np.int64),
        'plain_indices': np.array([0, 2], dtype=np.int64),
        'values': -np.arange(1, 13, dtype=np.float32).reshape(2, 2, 3),
        'size': [4, 9],
        'blocksize': (2, 3),
        'dtype': np.float32,
        'requires_grad': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Integer type (int64)
    input_dict_7 = {
        'compressed_indices': np.array([0, 2, 3], dtype=np.int64),
        'plain_indices': np.array([1, 3, 0], dtype=np.int64),
        'values': np.arange(24, dtype=np.int64).reshape(3, 4, 2),
        'size': [8, 8],
        'blocksize': (4, 2),
        'dtype': np.int64,
        'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: blocksize=(1, 1), equivalent to CSR
    input_dict_8 = {
        'compressed_indices': np.array([0, 1, 2, 2, 3, 4], dtype=np.int64),
        'plain_indices': np.array([0, 2, 1, 4], dtype=np.int64),
        'values': np.array([1, 2, 3, 4], dtype=np.float32).reshape(4, 1, 1),
        'size': [5, 5],
        'blocksize': (1, 1),
        'dtype': np.float32,
        'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: Blocksize where one dimension is the full tensor size
    input_dict_9 = {
        'compressed_indices': np.array([0, 2], dtype=np.int64),
        'plain_indices': np.array([1, 3], dtype=np.int64),
        'values': np.random.rand(2, 3, 2).astype(np.float64),
        'size': [3, 8],
        'blocksize': (3, 2),
        'dtype': np.float64,
        'requires_grad': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: Another complex pattern
    input_dict_10 = {
        'compressed_indices': np.array([0, 1, 3, 3, 4, 5], dtype=np.int64),
        'plain_indices': np.array([1, 0, 1, 0, 1], dtype=np.int64),
        'values': np.random.randn(5, 2, 5).astype(np.float32),
        'size': [10, 10],
        'blocksize': (2, 5),
        'dtype': np.float32,
        'requires_grad': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["torch.sparse_bsr_tensor_3"] = sparse_bsr_tensor_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.sparse_bsr_tensor_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sparse_bsr_tensor_3'.")

check_valid('torch.sparse_bsr_tensor', generated_inputs['torch.sparse_bsr_tensor_3'], lib="torch", suffix=3)
