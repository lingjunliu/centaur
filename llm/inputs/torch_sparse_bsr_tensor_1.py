
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def sparse_bsr_tensor_inputs():
    list_of_inputs = []

    # Input 1
    compressed_indices = torch.tensor(np.array([0, 1]))
    plain_indices = torch.tensor(np.array([[0, 0], [1, 1]]))
    values = torch.tensor(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32))
    size = (2, 2, 2, 2)
    blocksize = (2, 2)
    dtype = torch.float32
    requires_grad = False

    input_dict = {
        "compressed_indices": compressed_indices,
        "plain_indices": plain_indices,
        "values": values,
        "size": size,
        "blocksize": blocksize,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    compressed_indices = torch.tensor(np.array([0, 2]))
    plain_indices = torch.tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
    values = torch.tensor(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]],[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.int64))
    size = (2, 2, 2, 2)
    blocksize = (2, 2)
    dtype = torch.int64
    requires_grad = True
    input_dict = {
        "compressed_indices": compressed_indices,
        "plain_indices": plain_indices,
        "values": values,
        "size": size,
        "blocksize": blocksize,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    compressed_indices = torch.tensor(np.array([0, 1]))
    plain_indices = torch.tensor(np.array([[0, 0, 0], [1, 1, 1]]))
    values = torch.tensor(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64))
    size = (2, 2, 2, 2)
    blocksize = (2, 2)
    dtype = torch.float64
    requires_grad = False
    input_dict = {
        "compressed_indices": compressed_indices,
        "plain_indices": plain_indices,
        "values": values,
        "size": size,
        "blocksize": blocksize,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    compressed_indices = torch.tensor(np.array([0, 1, 2]))
    plain_indices = torch.tensor(np.array([[0, 0], [1, 1], [2, 2]]))
    values = torch.tensor(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32))
    size = (3, 2, 2, 2)
    blocksize = (2, 2)
    dtype = torch.float32
    requires_grad = True
    input_dict = {
        "compressed_indices": compressed_indices,
        "plain_indices": plain_indices,
        "values": values,
        "size": size,
        "blocksize": blocksize,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    compressed_indices = torch.tensor(np.array([0, 2, 3]))
    plain_indices = torch.tensor(np.array([[0, 0], [0, 1], [2, 0], [3, 1]]))
    values = torch.tensor(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.int32))
    size = (4, 2, 2, 2)
    blocksize = (2, 2)
    dtype = torch.int32
    requires_grad = False
    input_dict = {
        "compressed_indices": compressed_indices,
        "plain_indices": plain_indices,
        "values": values,
        "size": size,
        "blocksize": blocksize,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    compressed_indices = torch.tensor(np.array([0, 1]))
    plain_indices = torch.tensor(np.array([[0], [1]]))
    values = torch.tensor(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32))
    size = (2, 2, 2, 2)
    blocksize = (2, 2)
    dtype = torch.float32
    requires_grad = True

    input_dict = {
        "compressed_indices": compressed_indices,
        "plain_indices": plain_indices,
        "values": values,
        "size": size,
        "blocksize": blocksize,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    compressed_indices = torch.tensor(np.array([0]))
    plain_indices = torch.tensor(np.array([[0, 0]]))
    values = torch.tensor(np.array([[[1, 2], [3, 4]]], dtype=np.float32))
    size = (1, 2, 2, 2)
    blocksize = (2, 2)
    dtype = torch.float32
    requires_grad = False

    input_dict = {
        "compressed_indices": compressed_indices,
        "plain_indices": plain_indices,
        "values": values,
        "size": size,
        "blocksize": blocksize,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    compressed_indices = torch.tensor(np.array([0, 1]))
    plain_indices = torch.tensor(np.array([[0, 0], [1, 1]]))
    values = torch.tensor(np.array([[[1]], [[5]]], dtype=np.float32))
    size = (2, 1, 1, 1)
    blocksize = (1, 1)
    dtype = torch.float32
    requires_grad = False

    input_dict = {
        "compressed_indices": compressed_indices,
        "plain_indices": plain_indices,
        "values": values,
        "size": size,
        "blocksize": blocksize,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    compressed_indices = torch.tensor(np.array([0, 1]))
    plain_indices = torch.tensor(np.array([[0, 0], [1, 1]]))
    values = torch.tensor(np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]]], dtype=np.float32))
    size = (2, 3, 3, 3)
    blocksize = (3, 3)
    dtype = torch.float32
    requires_grad = False

    input_dict = {
        "compressed_indices": compressed_indices,
        "plain_indices": plain_indices,
        "values": values,
        "size": size,
        "blocksize": blocksize,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    compressed_indices = torch.tensor(np.array([0, 1]))
    plain_indices = torch.tensor(np.array([[0, 0], [1, 1]]))
    values = torch.tensor(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64))
    size = (2, 2, 2, 2)
    blocksize = (2, 2)
    dtype = torch.float64
    requires_grad = False
    input_dict = {
        "compressed_indices": compressed_indices,
        "plain_indices": plain_indices,
        "values": values,
        "size": size,
        "blocksize": blocksize,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.sparse_bsr_tensor_1"] = sparse_bsr_tensor_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sparse_bsr_tensor_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sparse_bsr_tensor_1'.")

check_valid('torch.sparse_bsr_tensor', generated_inputs['torch.sparse_bsr_tensor_1'], lib="torch", suffix=1)
