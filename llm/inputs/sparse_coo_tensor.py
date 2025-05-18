
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def sparse_coo_tensor_inputs():
    list_of_inputs = []

    indices = np.array([[0, 1], [1, 2]]).T
    values = np.array([1, 2], dtype=np.float32)
    size = (3, 4)

    input_dict = {
        "indices": indices,
        "values": values,
        "size": size,
        "dtype": np.float32,
        "requires_grad": False,
        "check_invariants": True,
        "is_coalesced": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = np.array([[0, 0], [1, 1], [2, 2]]).T
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    size = (3, 3)
    input_dict = {
        "indices": indices,
        "values": values,
        "size": size,
        "dtype": np.float32,
        "requires_grad": True,
        "check_invariants": True,
        "is_coalesced": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = np.array([[0, 1, 2], [2, 0, 1]]).T
    values = np.array([4, 5, 6], dtype=np.float64)
    size = (4, 4)
    input_dict = {
        "indices": indices,
        "values": values,
        "size": size,
        "dtype": np.float64,
        "requires_grad": False,
        "check_invariants": False,
        "is_coalesced": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = np.array([[0], [1], [2]]).T
    values = np.array([7, 8, 9], dtype=np.float16)
    size = (5,)
    input_dict = {
        "indices": indices,
        "values": values,
        "size": size,
        "dtype": np.float16,
        "requires_grad": True,
        "check_invariants": True,
        "is_coalesced": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = np.array([[0, 1, 2, 0], [1, 2, 0, 1]]).T
    values = np.array([10, 11, 12, 13], dtype=np.float32)
    size = (3, 3)
    input_dict = {
        "indices": indices,
        "values": values,
        "size": size,
        "dtype": np.float32,
        "requires_grad": False,
        "check_invariants": True,
        "is_coalesced": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

list_of_inputs = sparse_coo_tensor_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('sparse_coo_tensor', list_of_inputs)
