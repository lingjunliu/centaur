
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def sparse_csr_tensor_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D sparse tensor
    crow_indices = np.array([0, 2, 3, 4], dtype=np.int64)
    col_indices = np.array([0, 2, 2, 1], dtype=np.int64)
    values = np.array([1, 2, 3, 4], dtype=np.float32)
    size = (4, 4)
    dtype = torch.float32
    requires_grad = False
    input_dict = {
        "crow_indices": crow_indices,
        "col_indices": col_indices,
        "values": values,
        "size": size,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D sparse tensor
    crow_indices = np.array([0, 1, 2, 3], dtype=np.int64)
    col_indices = np.array([0, 1, 2], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.float64)
    size = (4,)
    dtype = torch.float64
    requires_grad = True
    input_dict = {
        "crow_indices": crow_indices,
        "col_indices": col_indices,
        "values": values,
        "size": size,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Empty sparse tensor
    crow_indices = np.array([0, 0, 0], dtype=np.int64)
    col_indices = np.array([], dtype=np.int64)
    values = np.array([], dtype=np.float32)
    size = (2, 2)
    dtype = torch.float32
    requires_grad = False
    input_dict = {
        "crow_indices": crow_indices,
        "col_indices": col_indices,
        "values": values,
        "size": size,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Larger sparse tensor with int values
    crow_indices = np.array([0, 3, 6, 8, 9], dtype=np.int64)
    col_indices = np.array([1, 3, 5, 0, 2, 4, 1, 5, 3], dtype=np.int64)
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)
    size = (5, 6)
    dtype = torch.int32
    requires_grad = False
    input_dict = {
        "crow_indices": crow_indices,
        "col_indices": col_indices,
        "values": values,
        "size": size,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Zero dimension
    crow_indices = np.array([0], dtype=np.int64)
    col_indices = np.array([], dtype=np.int64)
    values = np.array([], dtype=np.float32)
    size = (0, 0)
    dtype = torch.float32
    requires_grad = False
    input_dict = {
        "crow_indices": crow_indices,
        "col_indices": col_indices,
        "values": values,
        "size": size,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Int8
    crow_indices = np.array([0, 1, 2], dtype=np.int64)
    col_indices = np.array([0, 1], dtype=np.int64)
    values = np.array([1, 2], dtype=np.int8)
    size = (3, 3)
    dtype = torch.int8
    requires_grad = False
    input_dict = {
        "crow_indices": crow_indices,
        "col_indices": col_indices,
        "values": values,
        "size": size,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.sparse_csr_tensor_2"] = sparse_csr_tensor_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sparse_csr_tensor_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sparse_csr_tensor_2'.")

check_valid('torch.sparse_csr_tensor', generated_inputs['torch.sparse_csr_tensor_2'], lib="torch", suffix=2)
