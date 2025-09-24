
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def sparse_bsr_tensor_2_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D float tensor
    input_dict_1 = {
        'compressed_indices': np.array([0, 1, 2], dtype=np.int64),
        'plain_indices': np.array([1, 0], dtype=np.int64),
        'values': np.arange(8, dtype=np.float32).reshape(2, 2, 2),
        'size': (4, 4),
        'dtype': np.float32,
        'requires_grad': False,
        'layout': 'torch.sparse_bsr'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 3D batched tensor with float64 and requires_grad=True
    input_dict_2 = {
        'compressed_indices': np.array([[0, 1, 2, 3], [0, 2, 2, 3]], dtype=np.int64),
        'plain_indices': np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int64),
        'values': np.arange(24, dtype=np.float64).reshape(2, 3, 2, 2),
        'size': (2, 6, 4),
        'dtype': np.float64,
        'requires_grad': True,
        'layout': 'torch.sparse_bsr'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Integer type tensor
    input_dict_3 = {
        'compressed_indices': np.array([0, 2, 3], dtype=np.int64),
        'plain_indices': np.array([0, 1, 1], dtype=np.int64),
        'values': np.arange(12, dtype=np.int32).reshape(3, 2, 2),
        'size': (4, 4),
        'dtype': np.int32,
        'requires_grad': False,
        'layout': 'torch.sparse_bsr'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Empty tensor (0 non-zero blocks)
    input_dict_4 = {
        'compressed_indices': np.array([0, 0, 0, 0, 0, 0], dtype=np.int64),
        'plain_indices': np.array([], dtype=np.int64),
        'values': np.array([], dtype=np.float32).reshape(0, 2, 5),
        'size': (10, 10),
        'dtype': np.float32,
        'requires_grad': False,
        'layout': 'torch.sparse_bsr'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: "Dense" BSR tensor where all blocks are specified
    input_dict_5 = {
        'compressed_indices': np.array([0, 3, 6], dtype=np.int64),
        'plain_indices': np.array([0, 1, 2, 0, 1, 2], dtype=np.int64),
        'values': np.arange(24, dtype=np.float32).reshape(6, 2, 2),
        'size': (4, 6),
        'dtype': np.float32,
        'requires_grad': False,
        'layout': 'torch.sparse_bsr'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Non-square blocksize inferred from values
    input_dict_6 = {
        'compressed_indices': np.array([0, 2, 3, 4], dtype=np.int64),
        'plain_indices': np.array([0, 3, 1, 2], dtype=np.int64),
        'values': np.arange(24, dtype=np.float32).reshape(4, 3, 2),
        'size': (9, 8),
        'dtype': np.float32,
        'requires_grad': False,
        'layout': 'torch.sparse_bsr'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Complex type with requires_grad=True
    input_dict_7 = {
        'compressed_indices': np.array([0, 1, 2], dtype=np.int64),
        'plain_indices': np.array([1, 0], dtype=np.int64),
        'values': (np.arange(8).reshape(2, 2, 2) + 1j * np.arange(8, 16).reshape(2, 2, 2)).astype(np.complex128),
        'size': (4, 4),
        'dtype': np.complex128,
        'requires_grad': True,
        'layout': 'torch.sparse_bsr'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: blocksize=(1, 1), equivalent to CSR
    input_dict_8 = {
        'compressed_indices': np.array([0, 1, 2, 3, 4, 5], dtype=np.int64),
        'plain_indices': np.array([0, 1, 2, 3, 4], dtype=np.int64),
        'values': np.arange(1, 6, dtype=np.float32).reshape(5, 1, 1),
        'size': (5, 5),
        'dtype': np.float32,
        'requires_grad': True,
        'layout': 'torch.sparse_bsr'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Negative values with requires_grad=True
    input_dict_9 = {
        'compressed_indices': np.array([0, 1, 2], dtype=np.int64),
        'plain_indices': np.array([1, 0], dtype=np.int64),
        'values': np.arange(-8, 0, dtype=np.float32).reshape(2, 2, 2),
        'size': (4, 4),
        'dtype': np.float32,
        'requires_grad': True,
        'layout': 'torch.sparse_bsr'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Minimal valid tensor with int8 dtype
    input_dict_10 = {
        'compressed_indices': np.array([0, 1], dtype=np.int64),
        'plain_indices': np.array([0], dtype=np.int64),
        'values': np.array([[1, 2], [3, 4]], dtype=np.int8).reshape(1, 2, 2),
        'size': (2, 2),
        'dtype': np.int8,
        'requires_grad': False,
        'layout': 'torch.sparse_bsr'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["torch.sparse_bsr_tensor_2"] = sparse_bsr_tensor_2_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.sparse_bsr_tensor_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sparse_bsr_tensor_2'.")

check_valid('torch.sparse_bsr_tensor', generated_inputs['torch.sparse_bsr_tensor_2'], lib="torch", suffix=2)
