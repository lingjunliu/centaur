
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def sparse_bsr_tensor_inputs():
    generated_inputs = []

    # Input 1: Basic float tensor
    crow_indices = np.array([0, 2, 4])
    col_indices = np.array([0, 2, 1, 2])
    values = np.random.randn(4, 2, 2).astype(np.float32)
    size = (4, 4)
    blocksize = (2, 2)
    dtype = torch.float32
    requires_grad = False
    input_dict = {
        'crow_indices': crow_indices,
        'col_indices': col_indices,
        'values': values,
        'size': size,
        'blocksize': blocksize,
        'dtype': dtype,
        'requires_grad': requires_grad
    }
    generated_inputs.append(copy.deepcopy(input_dict))

    return generated_inputs

generated_inputs = sparse_bsr_tensor_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('sparse_bsr_tensor', generated_inputs)
