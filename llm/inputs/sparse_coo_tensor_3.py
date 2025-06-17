
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def sparse_coo_tensor_inputs():
    list_of_inputs = []

    # Example 1: Float tensor
    indices = np.array([[0, 1], [1, 2]], dtype=np.int64)
    values = np.array([3.0, 4.0], dtype=np.float32)
    size = (3, 4)
    input_dict = {
        "indices": indices,
        "values": values,
        "size": size,
        "dtype": torch.float32,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Int tensor
    indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.int64)
    size = (3, 4)
    input_dict = {
        "indices": indices,
        "values": values,
        "size": size,
        "dtype": torch.int64,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: Bool tensor
    indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    values = np.array([True, False], dtype=np.bool_)
    size = (2, 2)
    input_dict = {
        "indices": indices,
        "values": values,
        "size": size,
        "dtype": torch.bool,
        "requires_grad": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: Tensor with negative values
    indices = np.array([[0, 1], [1, 0]], dtype=np.int64)
    values = np.array([-1.5, 2.5], dtype=np.float64)
    size = (2, 2)
    input_dict = {
        "indices": indices,
        "values": values,
        "size": size,
        "dtype": torch.float64,
        "requires_grad": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5: 3D tensor
    indices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.float32)
    size = (2, 2, 2)
    input_dict = {
        "indices": indices,
        "values": values,
        "size": size,
        "dtype": torch.float32,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sparse_coo_tensor_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sparse_coo_tensor_3'.")

check_valid('torch.sparse_coo_tensor', generated_inputs['torch.sparse_coo_tensor_3'], lib="torch")
