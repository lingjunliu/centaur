
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def sparse_bsr_tensor_inputs():
    list_of_inputs = []

    # Input 1
    compressed_indices = torch.tensor([0, 1]).int()
    plain_indices = torch.tensor([[0, 0], [1, 1]]).int()
    values = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).float()
    size = [4, 4]
    blocksize = (2, 2)
    dtype = torch.float32
    requires_grad = False
    layout = "strided"

    input_dict = {
        'compressed_indices': compressed_indices,
        'plain_indices': plain_indices,
        'values': values,
        'size': size,
        'blocksize': blocksize,
        'dtype': dtype,
        'requires_grad': requires_grad,
        'layout': layout
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    compressed_indices = torch.tensor([0, 1, 2]).int()
    plain_indices = torch.tensor([[0, 0], [0, 1], [1, 0]]).int()
    values = torch.tensor([[[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]]]).float()
    size = [4, 4]
    blocksize = (2, 2)
    dtype = torch.float32
    requires_grad = True
    layout = "strided"

    input_dict = {
        'compressed_indices': compressed_indices,
        'plain_indices': plain_indices,
        'values': values,
        'size': size,
        'blocksize': blocksize,
        'dtype': dtype,
        'requires_grad': requires_grad,
        'layout': layout
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    compressed_indices = torch.tensor([0]).int()
    plain_indices = torch.tensor([[0, 0]]).int()
    values = torch.tensor([[[1, 2, 3], [4, 5, 6]]]).float()
    size = [2, 3]
    blocksize = (2, 3)
    dtype = torch.float32
    requires_grad = False
    layout = "strided"

    input_dict = {
        'compressed_indices': compressed_indices,
        'plain_indices': plain_indices,
        'values': values,
        'size': size,
        'blocksize': blocksize,
        'dtype': dtype,
        'requires_grad': requires_grad,
        'layout': layout
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    compressed_indices = torch.tensor([0, 1]).int()
    plain_indices = torch.tensor([[0, 0], [1, 2]]).int()
    values = torch.tensor([[[1, 1], [1, 1]], [[2, 2], [2, 2]]]).float()
    size = [4, 6]
    blocksize = (2, 2)
    dtype = torch.float32
    requires_grad = True
    layout = "strided"

    input_dict = {
        'compressed_indices': compressed_indices,
        'plain_indices': plain_indices,
        'values': values,
        'size': size,
        'blocksize': blocksize,
        'dtype': dtype,
        'requires_grad': requires_grad,
        'layout': layout
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    compressed_indices = torch.tensor([0, 1]).int()
    plain_indices = torch.tensor([[0, 0], [1, 1]]).int()
    values = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).float()
    size = [4, 4]
    blocksize = (2, 2)
    dtype = torch.float32
    requires_grad = False
    layout = "strided"

    input_dict = {
        'compressed_indices': compressed_indices,
        'plain_indices': plain_indices,
        'values': values,
        'size': size,
        'blocksize': blocksize,
        'dtype': dtype,
        'requires_grad': requires_grad,
        'layout': layout
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.sparse_bsr_tensor_4"] = sparse_bsr_tensor_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sparse_bsr_tensor_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sparse_bsr_tensor_4'.")

check_valid('torch.sparse_bsr_tensor', generated_inputs['torch.sparse_bsr_tensor_4'], lib="torch", suffix=4)
