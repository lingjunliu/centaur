
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy as np

def sparse_csr_tensor_inputs():
    list_of_inputs = []

    # Input 1: Basic 2x3 sparse matrix with float32 values
    input_dict_1 = {
        'crow_indices': torch.tensor([0, 2, 3], dtype=torch.int64).numpy(),
        'col_indices': torch.tensor([0, 2, 1], dtype=torch.int64).numpy(),
        'values': torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).numpy(),
        'size': (2, 3),
        'dtype': torch.float32,
        'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 3x3 sparse matrix with int64 values and an empty row
    input_dict_2 = {
        'crow_indices': torch.tensor([0, 1, 1, 3], dtype=torch.int64).numpy(),
        'col_indices': torch.tensor([1, 0, 2], dtype=torch.int64).numpy(),
        'values': torch.tensor([-10, 20, 30], dtype=torch.int64).numpy(),
        'size': (3, 3),
        'dtype': torch.int64,
        'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 2x2 sparse matrix with float64 values and requires_grad=True
    input_dict_3 = {
        'crow_indices': torch.tensor([0, 2, 2], dtype=torch.int64).numpy(),
        'col_indices': torch.tensor([0, 1], dtype=torch.int64).numpy(),
        'values': torch.tensor([1.1, 2.2], dtype=torch.float64).numpy(),
        'size': (2, 2),
        'dtype': torch.float64,
        'requires_grad': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Empty 4x5 sparse tensor
    input_dict_4 = {
        'crow_indices': torch.tensor([0, 0, 0, 0, 0], dtype=torch.int64).numpy(),
        'col_indices': torch.tensor([], dtype=torch.int64).numpy(),
        'values': torch.tensor([], dtype=torch.float32).numpy(),
        'size': (4, 5),
        'dtype': torch.float32,
        'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 1x5 sparse matrix (single row)
    input_dict_5 = {
        'crow_indices': torch.tensor([0, 3], dtype=torch.int64).numpy(),
        'col_indices': torch.tensor([0, 2, 4], dtype=torch.int64).numpy(),
        'values': torch.tensor([1, 2, 3], dtype=torch.int32).numpy(),
        'size': (1, 5),
        'dtype': torch.int32,
        'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: 5x1 sparse matrix (single column)
    input_dict_6 = {
        'crow_indices': torch.tensor([0, 1, 1, 2, 3, 3], dtype=torch.int64).numpy(),
        'col_indices': torch.tensor([0, 0, 0], dtype=torch.int64).numpy(),
        'values': torch.tensor([10., 20., 30.], dtype=torch.float32).numpy(),
        'size': (5, 1),
        'dtype': torch.float32,
        'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: Sparse matrix with a full row
    input_dict_7 = {
        'crow_indices': torch.tensor([0, 4, 4, 5], dtype=torch.int64).numpy(),
        'col_indices': torch.tensor([0, 1, 2, 3, 1], dtype=torch.int64).numpy(),
        'values': torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64).numpy(),
        'size': (3, 4),
        'dtype': torch.int64,
        'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: 3x3 identity matrix
    input_dict_8 = {
        'crow_indices': torch.tensor([0, 1, 2, 3], dtype=torch.int64).numpy(),
        'col_indices': torch.tensor([0, 1, 2], dtype=torch.int64).numpy(),
        'values': torch.tensor([1, 1, 1], dtype=torch.int32).numpy(),
        'size': (3, 3),
        'dtype': torch.int32,
        'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: A larger sparse matrix
    input_dict_9 = {
        'crow_indices': torch.tensor([0, 2, 3, 3, 3, 4, 6, 6, 7, 7, 8], dtype=torch.int64).numpy(),
        'col_indices': torch.tensor([5, 10, 8, 1, 2, 15, 19, 0], dtype=torch.int64).numpy(),
        'values': torch.tensor([1., 2., 3., 4., 5., 6., 7., 8.], dtype=torch.float32).numpy(),
        'size': (10, 20),
        'dtype': torch.float32,
        'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: float16 dtype
    input_dict_10 = {
        'crow_indices': torch.tensor([0, 1, 3], dtype=torch.int64).numpy(),
        'col_indices': torch.tensor([1, 0, 1], dtype=torch.int64).numpy(),
        'values': torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16).numpy(),
        'size': (2, 2),
        'dtype': torch.float16,
        'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["torch.sparse_csr_tensor_1"] = sparse_csr_tensor_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.sparse_csr_tensor_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sparse_csr_tensor_1'.")

check_valid('torch.sparse_csr_tensor', generated_inputs['torch.sparse_csr_tensor_1'], lib="torch", suffix=1)
