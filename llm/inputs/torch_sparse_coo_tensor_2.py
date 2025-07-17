
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def sparse_coo_tensor_inputs():
    list_of_inputs = []

    # Input 1
    indices = np.array([[0, 1], [1, 2]], dtype=np.int64).T
    values = np.array([1, 2], dtype=np.float32)
    size = (2, 3)
    dtype = np.float32
    requires_grad = False
    input_dict = {"indices": indices, "values": values, "size": size, "dtype": torch.float32, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    indices = np.array([[0, 0], [1, 1]], dtype=np.int64).T
    values = np.array([1.0, -2.0], dtype=np.float64)
    size = (3, 3)
    dtype = np.float64
    requires_grad = True
    input_dict = {"indices": indices, "values": values, "size": size, "dtype":  torch.float64, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    indices = np.array([[0], [2]], dtype=np.int64).T
    values = np.array([3, 4], dtype=np.int64)
    size = (5,)
    dtype = np.int64
    requires_grad = False
    input_dict = {"indices": indices, "values": values, "size": size, "dtype": torch.int64, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    indices = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int64).T
    values = np.array([5, 6, 7], dtype=np.int32)
    size = (3, 3)
    dtype = np.int32
    requires_grad = True
    input_dict = {"indices": indices, "values": values, "size": size, "dtype": torch.int32, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    indices = np.array([[0, 0, 1], [0, 1, 1], [0, 0, 0]], dtype=np.int64).T
    values = np.array([1, 2, 3], dtype=np.float16)
    size = (2, 2, 2)
    dtype = np.float16
    requires_grad = False
    input_dict = {"indices": indices, "values": values, "size": size, "dtype": torch.float16, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    indices = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int64).T
    values = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    size = (5, 5)
    dtype = np.float32
    requires_grad = True
    input_dict = {"indices": indices, "values": values, "size": size, "dtype": torch.float32, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    indices = np.array([[0, 0], [1, 2]], dtype=np.int64).T
    values = np.array([1, -1], dtype=np.int8)
    size = (2, 3)
    dtype = np.int8
    requires_grad = False
    input_dict = {"indices": indices, "values": values, "size": size, "dtype": torch.int8, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    indices = np.array([[0, 1], [1, 0]], dtype=np.int64).T
    values = np.array([1, 2], dtype=np.uint8)
    size = (4, 4)
    dtype = np.uint8
    requires_grad = True
    input_dict = {"indices": indices, "values": values, "size": size, "dtype": torch.uint8, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    indices = np.array([[0], [1]], dtype=np.int64).T
    values = np.array([1, 2], dtype=np.int16)
    size = (6,)
    dtype = np.int16
    requires_grad = False
    input_dict = {"indices": indices, "values": values, "size": size, "dtype": torch.int16, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    indices = np.array([[0, 1, 2], [1, 0, 1]], dtype=np.int64).T
    values = np.array([1.1, 2.2, 3.3], dtype=np.float64)
    size = (4, 4)
    dtype = np.float64
    requires_grad = True
    input_dict = {"indices": indices, "values": values, "size": size, "dtype": torch.float64, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.sparse_coo_tensor_2"] = sparse_coo_tensor_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sparse_coo_tensor_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sparse_coo_tensor_2'.")

check_valid('torch.sparse_coo_tensor', generated_inputs['torch.sparse_coo_tensor_2'], lib="torch", suffix=2)
