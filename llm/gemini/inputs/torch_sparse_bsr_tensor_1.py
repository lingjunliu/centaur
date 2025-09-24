
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def sparse_bsr_tensor_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D BSR tensor
    input_dict_1 = {
        'compressed_indices': np.array([0, 1, 2, 3], dtype=np.int64),
        'plain_indices': np.array([1, 0, 1], dtype=np.int64),
        'values': np.arange(1, 13, dtype=np.float32).reshape(3, 2, 2),
        'size': (6, 4),
        'dtype': np.float32,
        'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: A row with multiple non-zero blocks, float64, requires_grad=True
    input_dict_2 = {
        'compressed_indices': np.array([0, 2, 3], dtype=np.int64),
        'plain_indices': np.array([0, 2, 1], dtype=np.int64),
        'values': np.random.rand(3, 3, 2).astype(np.float64),
        'size': (6, 6),
        'dtype': np.float64,
        'requires_grad': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Integer type with an empty block-row
    input_dict_3 = {
        'compressed_indices': np.array([0, 1, 1, 3, 4], dtype=np.int64),
        'plain_indices': np.array([1, 0, 1, 0], dtype=np.int64),
        'values': np.arange(1, 17, dtype=np.int32).reshape(4, 2, 2),
        'size': (8, 4),
        'dtype': np.int32,
        'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: An empty BSR tensor
    input_dict_4 = {
        'compressed_indices': np.array([0, 0, 0], dtype=np.int64),
        'plain_indices': np.array([], dtype=np.int64),
        'values': np.empty((0, 2, 3), dtype=np.float32),
        'size': (4, 6),
        'dtype': np.float32,
        'requires_grad': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: A "dense" BSR tensor (all blocks are non-zero) with complex numbers
    input_dict_5 = {
        'compressed_indices': np.array([0, 2, 4], dtype=np.int64),
        'plain_indices': np.array([0, 1, 0, 1], dtype=np.int64),
        'values': (np.arange(16, dtype=np.float32).reshape(4, 2, 2) +
                   1j * np.arange(16, 32, dtype=np.float32).reshape(4, 2, 2)),
        'size': (4, 4),
        'dtype': np.complex64,
        'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: A 3D (batched) BSR tensor
    input_dict_6 = {
        'compressed_indices': np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int64),
        'plain_indices': np.array([[0, 1], [1, 0]], dtype=np.int64),
        'values': np.random.rand(2, 2, 2, 3).astype(np.float64),
        'size': (2, 4, 6),
        'dtype': np.float64,
        'requires_grad': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Batched BSR with empty block-rows
    input_dict_7 = {
        'compressed_indices': np.array([[0, 1, 1, 2], [0, 0, 2, 2]], dtype=np.int64),
        'plain_indices': np.array([[0, 1], [0, 1]], dtype=np.int64),
        'values': np.ones((2, 2, 2, 2), dtype=np.int16),
        'size': (2, 6, 4),
        'dtype': np.int16,
        'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Minimal non-empty tensor with 1x1 blocks (equivalent to CSR)
    input_dict_8 = {
        'compressed_indices': np.array([0, 1], dtype=np.int64),
        'plain_indices': np.array([0], dtype=np.int64),
        'values': np.array([[[5]]], dtype=np.int8),
        'size': (1, 1),
        'dtype': np.int8,
        'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Using negative values
    input_dict_9 = {
        'compressed_indices': np.array([0, 1, 2, 3], dtype=np.int64),
        'plain_indices': np.array([1, 0, 1], dtype=np.int64),
        'values': -np.arange(1, 13, dtype=np.float32).reshape(3, 2, 2),
        'size': (6, 4),
        'dtype': np.float32,
        'requires_grad': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Complex numbers with requires_grad=True
    input_dict_10 = {
        'compressed_indices': np.array([0, 1, 2], dtype=np.int64),
        'plain_indices': np.array([1, 0], dtype=np.int64),
        'values': (np.random.rand(2, 2, 3) + 1j * np.random.rand(2, 2, 3)).astype(np.complex128),
        'size': (4, 6),
        'dtype': np.complex128,
        'requires_grad': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: A 4D BSR tensor (2 batch dimensions)
    input_dict_11 = {
        'compressed_indices': np.broadcast_to(np.array([0, 1, 2], dtype=np.int64), (2, 2, 3)),
        'plain_indices': np.array([[[0, 1], [1, 0]], [[1, 1], [0, 0]]], dtype=np.int64),
        'values': np.random.rand(2, 2, 2, 2, 2).astype(np.float32),
        'size': (2, 2, 4, 4),
        'dtype': np.float32,
        'requires_grad': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.sparse_bsr_tensor_1"] = sparse_bsr_tensor_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.sparse_bsr_tensor_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sparse_bsr_tensor_1'.")

check_valid('torch.sparse_bsr_tensor', generated_inputs['torch.sparse_bsr_tensor_1'], lib="torch", suffix=1)
