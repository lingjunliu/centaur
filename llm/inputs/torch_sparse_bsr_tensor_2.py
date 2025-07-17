
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def sparse_bsr_tensor_inputs():
    list_of_inputs = []

    def create_input_dict(compressed_indices, plain_indices, values, size, blocksize, dtype, requires_grad, layout):
        return {
            "compressed_indices": torch.tensor(compressed_indices),
            "plain_indices": torch.tensor(plain_indices),
            "values": torch.tensor(values),
            "size": size,
            "blocksize": blocksize,
            "dtype": dtype,
            "requires_grad": requires_grad,
            "layout": layout
        }

    # Input 1
    compressed_indices = np.array([0, 1])
    plain_indices = np.array([[0, 0], [1, 1]])
    values = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    size = (2, 2, 2, 2)
    blocksize = (2, 2)
    dtype = torch.float32
    requires_grad = False
    layout = "torch.sparse_bsr"
    list_of_inputs.append(copy.deepcopy(create_input_dict(compressed_indices, plain_indices, values, size, blocksize, dtype, requires_grad, layout)))

    # Input 2
    compressed_indices = np.array([0, 2])
    plain_indices = np.array([[0, 0], [1, 0]])
    values = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64)
    size = (2, 1, 2, 2)
    blocksize = (2, 2)
    dtype = torch.float64
    requires_grad = True
    layout = "torch.sparse_bsr"
    list_of_inputs.append(copy.deepcopy(create_input_dict(compressed_indices, plain_indices, values, size, blocksize, dtype, requires_grad, layout)))

    # Input 3
    compressed_indices = np.array([0, 1, 2])
    plain_indices = np.array([[0, 0], [0, 1], [1, 0]])
    values = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.int32)
    size = (2, 2, 2, 2)
    blocksize = (2, 2)
    dtype = torch.int32
    requires_grad = False
    layout = "torch.sparse_bsr"
    list_of_inputs.append(copy.deepcopy(create_input_dict(compressed_indices, plain_indices, values, size, blocksize, dtype, requires_grad, layout)))

    # Input 4
    compressed_indices = np.array([0, 1])
    plain_indices = np.array([[0, 0], [1, 1]])
    values = np.array([[[1, 2]], [[5, 6]]], dtype=np.float32)
    size = (2, 2, 1, 2)
    blocksize = (1, 2)
    dtype = torch.float32
    requires_grad = True
    layout = "torch.sparse_bsr"
    list_of_inputs.append(copy.deepcopy(create_input_dict(compressed_indices, plain_indices, values, size, blocksize, dtype, requires_grad, layout)))

    # Input 5
    compressed_indices = np.array([0])
    plain_indices = np.array([[0, 0]])
    values = np.array([[[1, 2], [3, 4]]], dtype=np.float32)
    size = (1, 1, 2, 2)
    blocksize = (2, 2)
    dtype = torch.float32
    requires_grad = False
    layout = "torch.sparse_bsr"
    list_of_inputs.append(copy.deepcopy(create_input_dict(compressed_indices, plain_indices, values, size, blocksize, dtype, requires_grad, layout)))

    # Input 6 - Removed int64
    compressed_indices = np.array([0, 1])
    plain_indices = np.array([[0, 0], [0, 1]])
    values = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float32)
    size = (1, 2, 2, 3)
    blocksize = (2, 3)
    dtype = torch.float32
    requires_grad = True
    layout = "torch.sparse_bsr"
    list_of_inputs.append(copy.deepcopy(create_input_dict(compressed_indices, plain_indices, values, size, blocksize, dtype, requires_grad, layout)))

    # Input 7
    compressed_indices = np.array([0])
    plain_indices = np.array([[0, 0]])
    values = np.array([[[1]]], dtype=np.float32)
    size = (1, 1, 1, 1)
    blocksize = (1, 1)
    dtype = torch.float32
    requires_grad = False
    layout = "torch.sparse_bsr"
    list_of_inputs.append(copy.deepcopy(create_input_dict(compressed_indices, plain_indices, values, size, blocksize, dtype, requires_grad, layout)))

    # Input 8
    compressed_indices = np.array([0, 1])
    plain_indices = np.array([[0, 0], [1, 0]])
    values = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]], dtype=np.float64)
    size = (2, 1, 2, 4)
    blocksize = (2, 4)
    dtype = torch.float64
    requires_grad = True
    layout = "torch.sparse_bsr"
    list_of_inputs.append(copy.deepcopy(create_input_dict(compressed_indices, plain_indices, values, size, blocksize, dtype, requires_grad, layout)))

    # Input 9
    compressed_indices = np.array([0, 1, 2])
    plain_indices = np.array([[0, 0], [0, 1], [1, 0]])
    values = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.int8)
    size = (2, 2, 2, 2)
    blocksize = (2, 2)
    dtype = torch.int8
    requires_grad = False
    layout = "torch.sparse_bsr"
    list_of_inputs.append(copy.deepcopy(create_input_dict(compressed_indices, plain_indices, values, size, blocksize, dtype, requires_grad, layout)))

    # Input 10
    compressed_indices = np.array([0, 1])
    plain_indices = np.array([[0, 0], [1, 1]])
    values = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint8)
    size = (2, 2, 2, 2)
    blocksize = (2, 2)
    dtype = torch.uint8
    requires_grad = True
    layout = "torch.sparse_bsr"
    list_of_inputs.append(copy.deepcopy(create_input_dict(compressed_indices, plain_indices, values, size, blocksize, dtype, requires_grad, layout)))

    # Input 11
    compressed_indices = np.array([0, 1])
    plain_indices = np.array([[0, 0], [1, 1]])
    values = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    size = (2, 2, 2, 2)
    blocksize = (2, 2)
    dtype = torch.float32
    requires_grad = False
    layout = "torch.sparse_bsr"
    list_of_inputs.append(copy.deepcopy(create_input_dict(compressed_indices, plain_indices, values, size, blocksize, dtype, requires_grad, layout)))
    
    # Input 12. Reduced size and number of indices.
    compressed_indices = np.array([0])
    plain_indices = np.array([[0, 0]])
    values = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    size = (1, 1, 2, 2)
    blocksize = (2, 2)
    dtype = torch.float32
    requires_grad = False
    layout = "torch.sparse_bsr"
    list_of_inputs.append(copy.deepcopy(create_input_dict(compressed_indices, plain_indices, values, size, blocksize, dtype, requires_grad, layout)))

    return list_of_inputs

generated_inputs["torch.sparse_bsr_tensor_2"] = sparse_bsr_tensor_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sparse_bsr_tensor_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sparse_bsr_tensor_2'.")

check_valid('torch.sparse_bsr_tensor', generated_inputs['torch.sparse_bsr_tensor_2'], lib="torch", suffix=2)
