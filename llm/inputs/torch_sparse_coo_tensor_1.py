
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def sparse_coo_tensor_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D sparse tensor
    indices = np.array([[0, 1], [1, 2]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.float32)
    size = (3, 4)
    dtype = np.float32
    requires_grad = False
    input_dict = {"indices": torch.tensor(indices), "values": torch.tensor(values, dtype=torch.float32), "size": size, "dtype": dtype, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D sparse tensor
    indices = np.array([[0], [2], [4]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.int64)
    size = (5,)
    dtype = np.int64
    requires_grad = True
    input_dict = {"indices": torch.tensor(indices), "values": torch.tensor(values, dtype=torch.float32), "size": size, "dtype": dtype, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D sparse tensor
    indices = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.float64)
    size = (3, 3, 3)
    dtype = np.float64
    requires_grad = False
    input_dict = {"indices": torch.tensor(indices), "values": torch.tensor(values, dtype=torch.float64), "size": size, "dtype": dtype, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty sparse tensor
    indices = np.empty((0, 2), dtype=np.int64)
    values = np.empty((0,), dtype=np.float32)
    size = (3, 4)
    dtype = np.float32
    requires_grad = False
    input_dict = {"indices": torch.tensor(indices), "values": torch.tensor(values, dtype=torch.float32), "size": size, "dtype": dtype, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different dtype
    indices = np.array([[0, 1], [1, 2]], dtype=np.int64)
    values = np.array([True, False], dtype=np.bool_)
    size = (3, 4)
    dtype = np.bool_
    requires_grad = False
    input_dict = {"indices": torch.tensor(indices), "values": torch.tensor(values), "size": size, "dtype": dtype, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: requires_grad = True
    indices = np.array([[0, 1], [1, 2]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.float32)
    size = (3, 4)
    dtype = np.float32
    requires_grad = True
    input_dict = {"indices": torch.tensor(indices), "values": torch.tensor(values, dtype=torch.float32), "size": size, "dtype": dtype, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger size
    indices = np.array([[0, 1], [1, 2], [5, 7]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.float32)
    size = (10, 10)
    dtype = np.float32
    requires_grad = False
    input_dict = {"indices": torch.tensor(indices), "values": torch.tensor(values, dtype=torch.float32), "size": size, "dtype": dtype, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Float64 values
    indices = np.array([[0, 1], [1, 2]], dtype=np.int64)
    values = np.array([1.5, 2.5], dtype=np.float64)
    size = (3, 4)
    dtype = np.float64
    requires_grad = False
    input_dict = {"indices": torch.tensor(indices), "values": torch.tensor(values, dtype=torch.float64), "size": size, "dtype": dtype, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Int16 values
    indices = np.array([[0, 1], [1, 2]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.int16)
    size = (3, 4)
    dtype = np.int16
    requires_grad = False
    input_dict = {"indices": torch.tensor(indices), "values": torch.tensor(values, dtype=torch.float32), "size": size, "dtype": dtype, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Unsorted indices
    indices = np.array([[1, 2], [0, 1]], dtype=np.int64)
    values = np.array([2, 1], dtype=np.float32)
    size = (3, 4)
    dtype = np.float32
    requires_grad = False
    input_dict = {"indices": torch.tensor(indices), "values": torch.tensor(values, dtype=torch.float32), "size": size, "dtype": dtype, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.sparse_coo_tensor_1"] = sparse_coo_tensor_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sparse_coo_tensor_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sparse_coo_tensor_1'.")

check_valid('torch.sparse_coo_tensor', generated_inputs['torch.sparse_coo_tensor_1'], lib="torch", suffix=1)
