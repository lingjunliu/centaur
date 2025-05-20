
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def sparse_bsr_tensor_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    crow_indices = np.array([0, 2]).astype(np.int64)
    col_indices = np.array([0, 1]).astype(np.int64)
    values = np.random.randn(2, 2, 2).astype(np.float32)
    size = (4, 4)
    blocksize = (2, 2)
    dtype = torch.float32
    requires_grad = False
    
    input_dict = {
        "crow_indices": crow_indices,
        "col_indices": col_indices,
        "values": values,
        "size": size,
        "blocksize": blocksize,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Int tensor with different blocksize
    crow_indices = np.array([0, 1]).astype(np.int64)
    col_indices = np.array([0]).astype(np.int64)
    values = np.random.randint(0, 10, size=(1, 2, 3)).astype(np.int64)
    size = (2, 3)
    blocksize = (2, 3)
    dtype = torch.int64
    requires_grad = True
    
    input_dict = {
        "crow_indices": crow_indices,
        "col_indices": col_indices,
        "values": values,
        "size": size,
        "blocksize": blocksize,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = sparse_bsr_tensor_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('sparse_bsr_tensor', generated_inputs)
