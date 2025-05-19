
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def sparse_coo_tensor_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D COO tensor with integer values
    indices = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.int64)
    size = (3, 3)
    input_dict = {
        "indices": indices,
        "values": values,
        "size": size,
        "dtype": np.int64,
        "requires_grad": False,
        "check_invariants": True,
        "is_coalesced": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D COO tensor with float values
    indices = np.array([[0, 0, 1], [1, 2, 0]], dtype=np.int64)
    values = np.array([1.5, 2.5], dtype=np.float32)
    size = (2, 3, 2)
    input_dict = {
        "indices": indices,
        "values": values,
        "size": size,
        "dtype": np.float32,
        "requires_grad": True,
        "check_invariants": False,
        "is_coalesced": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D COO tensor with complex values
    indices = np.array([[0, 1], [1, 0]], dtype=np.int64)
    values = np.array([1 + 1j, 2 - 2j], dtype=np.complex64)
    size = (2, 2)
    input_dict = {
        "indices": indices,
        "values": values,
        "size": size,
        "dtype": np.complex64,
        "requires_grad": False,
        "check_invariants": True,
        "is_coalesced": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = sparse_coo_tensor_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('sparse_coo_tensor', generated_inputs)
