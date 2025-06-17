
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def sparse_bsr_tensor_3_inputs():
    list_of_inputs = []

    # Input 1: Basic example with float values
    compressed_indices = np.array([0, 1, 2])
    plain_indices = np.array([[0, 0], [1, 0], [2, 0]])
    values = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], [[9.0, 10.0], [11.0, 12.0]]])
    size = [6, 2]
    blocksize = (2, 2)
    dtype = torch.float32
    requires_grad = False
    
    compressed_indices_tensor = torch.tensor(compressed_indices)
    plain_indices_tensor = torch.tensor(plain_indices, dtype=torch.long)
    values_tensor = torch.tensor(values, dtype=dtype)
    
    input_dict = {
        "crow_indices": compressed_indices_tensor,
        "col_indices": plain_indices_tensor,
        "values": values_tensor,
        "size": size,
        "blocksize": blocksize,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer values
    compressed_indices = np.array([0, 1])
    plain_indices = np.array([[0, 0], [1, 1]])
    values = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    size = [4, 4]
    blocksize = (2, 2)
    dtype = torch.int64
    requires_grad = False
    
    compressed_indices_tensor = torch.tensor(compressed_indices)
    plain_indices_tensor = torch.tensor(plain_indices, dtype=torch.long)
    values_tensor = torch.tensor(values, dtype=dtype)
    
    input_dict = {
        "crow_indices": compressed_indices_tensor,
        "col_indices": plain_indices_tensor,
        "values": values_tensor,
        "size": size,
        "blocksize": blocksize,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different blocksize
    compressed_indices = np.array([0, 1])
    plain_indices = np.array([[0, 0], [2, 2]])
    values = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]])
    size = [5, 5]
    blocksize = (2, 3)
    dtype = torch.float64
    requires_grad = True

    compressed_indices_tensor = torch.tensor(compressed_indices)
    plain_indices_tensor = torch.tensor(plain_indices, dtype=torch.long)
    values_tensor = torch.tensor(values, dtype=dtype)
    
    input_dict = {
        "crow_indices": compressed_indices_tensor,
        "col_indices": plain_indices_tensor,
        "values": values_tensor,
        "size": size,
        "blocksize": blocksize,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.sparse_bsr_tensor_3"] = sparse_bsr_tensor_3_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sparse_bsr_tensor_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sparse_bsr_tensor_3'.")

check_valid('torch.sparse_bsr_tensor', generated_inputs['torch.sparse_bsr_tensor_3'], lib="torch")
