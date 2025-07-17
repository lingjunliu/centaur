
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def sparse_bsr_tensor_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D example
    compressed_indices = torch.tensor(np.array([0, 1], dtype=np.int32))
    plain_indices = torch.tensor(np.array([[0, 0], [1, 1]]))
    values = torch.tensor(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    size = (2, 2)
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

    # Input 2: requires_grad = True
    compressed_indices = torch.tensor(np.array([0, 1], dtype=np.int32))
    plain_indices = torch.tensor(np.array([[0, 0], [1, 1]]))
    values = torch.tensor(np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]))
    size = (2, 2)
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

    # Input 3: Different blocksize
    compressed_indices = torch.tensor(np.array([0, 1], dtype=np.int32))
    plain_indices = torch.tensor(np.array([[0, 0], [0, 1]]))
    values = torch.tensor(np.array([[[1, 2, 3]], [[4, 5, 6]]]))
    size = (1, 2)
    blocksize = (1, 3)
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

    # Input 4: Larger size, more blocks
    compressed_indices = torch.tensor(np.array([0, 1, 2], dtype=np.int32))
    plain_indices = torch.tensor(np.array([[0, 0], [0, 1], [1, 0]]))
    values = torch.tensor(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]))
    size = (2, 2)
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

    # Input 5:  1D compressed_indices, and plain_indices
    compressed_indices = torch.tensor(np.array([0, 1], dtype=np.int32))
    plain_indices = torch.tensor(np.array([[0], [2]]))
    values = torch.tensor(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    size = (3, 2)
    blocksize = (1, 2)
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

    # Input 6: different shape of values (3d), while blocksize = (1,1)
    compressed_indices = torch.tensor(np.array([0, 1], dtype=np.int32))
    plain_indices = torch.tensor(np.array([[0, 0], [1, 1]]))
    values = torch.tensor(np.array([[[1]], [[2]]]))
    size = (2, 2)
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
    
    # Input 7: Smaller values
    compressed_indices = torch.tensor(np.array([0], dtype=np.int32))
    plain_indices = torch.tensor(np.array([[0, 0]]))
    values = torch.tensor(np.array([[[0.1, 0.2], [0.3, 0.4]]]))
    size = (1, 1)
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

    # Input 8: Larger plain_indices
    compressed_indices = torch.tensor(np.array([0, 1], dtype=np.int32))
    plain_indices = torch.tensor(np.array([[0, 1], [2, 3]]))
    values = torch.tensor(np.array([[[1, 2]], [[5, 6]]]))
    size = (3, 4)
    blocksize = (1, 2)
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

    # Input 9: Different blocksize
    compressed_indices = torch.tensor(np.array([0, 1], dtype=np.int32))
    plain_indices = torch.tensor(np.array([[0, 0], [1, 1]]))
    values = torch.tensor(np.array([[[1, 2, 3], [4,5,6]], [[7, 8, 9], [10, 11, 12]]]))
    size = (2, 2)
    blocksize = (2, 3)
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
    
    #Input 10: Identity
    compressed_indices = torch.tensor(np.array([0], dtype=np.int32))
    plain_indices = torch.tensor(np.array([[0,0]]))
    values = torch.tensor(np.array([[[1.0, 0.0], [0.0, 1.0]]]))
    size = (2, 2)
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
