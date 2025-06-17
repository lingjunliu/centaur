
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def sparse_bsr_tensor_inputs():
    list_of_inputs = []

    # Input 1: Basic example with float values
    compressed_indices = np.array([0, 2]).astype(np.int64)
    plain_indices = np.array([[0, 0], [1, 1]]).astype(np.int64)
    values = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).astype(np.float32)
    size = (2, 2)
    blocksize = (1, 1)

    input_dict = {
        "compressed_indices": torch.tensor(compressed_indices),
        "plain_indices": torch.tensor(plain_indices),
        "values": torch.tensor(values),
        "size": size,
        "blocksize": blocksize,
        "dtype": torch.float32,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer values, different block size
    compressed_indices = np.array([0, 1]).astype(np.int64)
    plain_indices = np.array([[0, 0], [1, 0]]).astype(np.int64)
    values = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
                       [[17, 18, 19, 20], [21, 22, 23, 24], [25, 26, 27, 28], [29, 30, 31, 32]]]).astype(np.int32)
    size = (2, 1)
    blocksize = (2, 2)

    input_dict = {
        "compressed_indices": torch.tensor(compressed_indices),
        "plain_indices": torch.tensor(plain_indices),
        "values": torch.tensor(values),
        "size": size,
        "blocksize": blocksize,
        "dtype": torch.int32,
        "requires_grad": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different size and plain indices
    compressed_indices = np.array([0, 2]).astype(np.int64)
    plain_indices = np.array([[0, 0], [0, 2]]).astype(np.int64)
    values = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).astype(np.float64)
    size = (2, 3)
    blocksize = (1, 1)

    input_dict = {
        "compressed_indices": torch.tensor(compressed_indices),
        "plain_indices": torch.tensor(plain_indices),
        "values": torch.tensor(values),
        "size": size,
        "blocksize": blocksize,
        "dtype": torch.float64,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4:  Blocksize (2, 1)
    compressed_indices = np.array([0, 1]).astype(np.int64)
    plain_indices = np.array([[0, 0], [1, 0]]).astype(np.int64)
    values = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).astype(np.int64)
    size = (2, 1)
    blocksize = (2, 1)

    input_dict = {
        "compressed_indices": torch.tensor(compressed_indices),
        "plain_indices": torch.tensor(plain_indices),
        "values": torch.tensor(values),
        "size": size,
        "blocksize": blocksize,
        "dtype": torch.int64,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative values
    compressed_indices = np.array([0, 1]).astype(np.int64)
    plain_indices = np.array([[0, 0], [1, 0]]).astype(np.int64)
    values = np.array([[[1, -2], [-3, 4]], [[-5, 6], [7, -8]]]).astype(np.int64)
    size = (2, 1)
    blocksize = (2, 1)

    input_dict = {
        "compressed_indices": torch.tensor(compressed_indices),
        "plain_indices": torch.tensor(plain_indices),
        "values": torch.tensor(values),
        "size": size,
        "blocksize": blocksize,
        "dtype": torch.int64,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.sparse_bsr_tensor_2"] = sparse_bsr_tensor_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sparse_bsr_tensor_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sparse_bsr_tensor_2'.")

check_valid('torch.sparse_bsr_tensor', generated_inputs['torch.sparse_bsr_tensor_2'], lib="torch")
