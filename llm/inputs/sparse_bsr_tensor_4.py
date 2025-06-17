
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def sparse_bsr_tensor_inputs():
    list_of_inputs = []

    # Input 1: Basic example with float values
    compressed_indices = np.array([0, 2])
    plain_indices = np.array([[0, 0], [1, 1]])
    values = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    size = (4, 4)
    blocksize = (2, 2)
    dtype = torch.float32
    requires_grad = False
    layout = torch.sparse_bsr

    input_dict = {
        "compressed_indices": torch.tensor(compressed_indices),
        "plain_indices": torch.tensor(plain_indices, dtype=torch.int64),
        "values": torch.tensor(values),
        "size": size,
        "blocksize": blocksize,
        "dtype": dtype,
        "requires_grad": requires_grad,
        "layout": layout
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer values
    compressed_indices = np.array([0, 1])
    plain_indices = np.array([[0, 0]])
    values = np.array([[[1, 2], [3, 4]]])
    size = (2, 2)
    blocksize = (2, 2)
    dtype = torch.int32
    requires_grad = False
    layout = torch.sparse_bsr

    input_dict = {
        "compressed_indices": torch.tensor(compressed_indices),
        "plain_indices": torch.tensor(plain_indices, dtype=torch.int64),
        "values": torch.tensor(values),
        "size": size,
        "blocksize": blocksize,
        "dtype": dtype,
        "requires_grad": requires_grad,
        "layout": layout
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative values and different dimensions
    compressed_indices = np.array([0, 2])
    plain_indices = np.array([[0, 0], [1, 1]])
    values = np.array([[[1.0, -2.0], [-3.0, 4.0]], [[-5.0, 6.0], [7.0, -8.0]]])
    size = (4, 4)
    blocksize = (2, 2)
    dtype = torch.float64
    requires_grad = True
    layout = torch.sparse_bsr

    input_dict = {
        "compressed_indices": torch.tensor(compressed_indices),
        "plain_indices": torch.tensor(plain_indices, dtype=torch.int64),
        "values": torch.tensor(values),
        "size": size,
        "blocksize": blocksize,
        "dtype": dtype,
        "requires_grad": requires_grad,
        "layout": layout
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different blocksize and size
    compressed_indices = np.array([0, 1])
    plain_indices = np.array([[0, 0]])
    values = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
    size = (3, 3)
    blocksize = (3, 3)
    dtype = torch.float32
    requires_grad = False
    layout = torch.sparse_bsr

    input_dict = {
        "compressed_indices": torch.tensor(compressed_indices),
        "plain_indices": torch.tensor(plain_indices, dtype=torch.int64),
        "values": torch.tensor(values),
        "size": size,
        "blocksize": blocksize,
        "dtype": dtype,
        "requires_grad": requires_grad,
        "layout": layout
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Another example with int values and different size
    compressed_indices = np.array([0, 1])
    plain_indices = np.array([[0, 0], [0, 1]])
    values = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    size = (4, 4)
    blocksize = (2, 2)
    dtype = torch.int64
    requires_grad = False
    layout = torch.sparse_bsr

    input_dict = {
        "compressed_indices": torch.tensor(compressed_indices),
        "plain_indices": torch.tensor(plain_indices, dtype=torch.int64),
        "values": torch.tensor(values),
        "size": size,
        "blocksize": blocksize,
        "dtype": dtype,
        "requires_grad": requires_grad,
        "layout": layout
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.sparse_bsr_tensor_4"] = sparse_bsr_tensor_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sparse_bsr_tensor_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sparse_bsr_tensor_4'.")

check_valid('torch.sparse_bsr_tensor', generated_inputs['torch.sparse_bsr_tensor_4'], lib="torch")
