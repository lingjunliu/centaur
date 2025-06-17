
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def sparse_bsr_tensor_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    compressed_indices = np.array([0, 1]).astype(np.int64)
    plain_indices = np.array([[0, 0], [1, 1]]).astype(np.int64)
    values = np.random.randn(2, 2, 2).astype(np.float32)
    size = (4, 4)
    blocksize = (2, 2)
    dtype = torch.float32
    requires_grad = False

    input_dict = {
        'compressed_indices': compressed_indices,
        'plain_indices': plain_indices,
        'values': values,
        'size': size,
        'blocksize': blocksize
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Int tensor, different size and blocksize
    compressed_indices = np.array([0, 1]).astype(np.int64)
    plain_indices = np.array([[0, 0], [0, 1]]).astype(np.int64)
    values = np.random.randint(0, 10, size=(2, 1, 3)).astype(np.int64)
    size = (2, 6)
    blocksize = (1, 3)
    dtype = torch.int64
    requires_grad = True

    input_dict = {
        'compressed_indices': compressed_indices,
        'plain_indices': plain_indices,
        'values': values,
        'size': size,
        'blocksize': blocksize
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative values, different dtype
    compressed_indices = np.array([0]).astype(np.int64)
    plain_indices = np.array([[0, 0]]).astype(np.int64)
    values = np.random.randn(1, 3, 1) * -1.0  # Negative values
    values = values.astype(np.float64)
    size = (3, 1)
    blocksize = (3, 1)
    dtype = torch.float64
    requires_grad = False
    input_dict = {
        'compressed_indices': compressed_indices,
        'plain_indices': plain_indices,
        'values': values,
        'size': size,
        'blocksize': blocksize
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: bool
    compressed_indices = np.array([0]).astype(np.int64)
    plain_indices = np.array([[0, 0]]).astype(np.int64)
    values = np.array([[[True, False], [False, True]]]).astype(np.bool_)
    size = (2, 2)
    blocksize = (2, 2)
    dtype = torch.bool
    requires_grad = False

    input_dict = {
        'compressed_indices': compressed_indices,
        'plain_indices': plain_indices,
        'values': values,
        'size': size,
        'blocksize': blocksize
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.sparse_bsr_tensor_1"] = sparse_bsr_tensor_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sparse_bsr_tensor_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sparse_bsr_tensor_1'.")

check_valid('torch.sparse_bsr_tensor', generated_inputs['torch.sparse_bsr_tensor_1'], lib="torch")
