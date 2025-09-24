
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def spmm_inputs():
    list_of_inputs = []

    # Input 1: Basic example with small sparse matrix and dense matrix
    indices = torch.tensor([[0, 1], [1, 0], [2, 2]])
    values = torch.tensor([1.0, 2.0, 3.0])
    size = (3, 3)
    input = torch.sparse_coo_tensor(indices, values, size, requires_grad=False).to_dense().numpy()
    mat2 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).numpy()

    input_dict = {
        "input": input,
        "mat2": mat2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Larger matrices
    indices = torch.tensor([[0, 0, 1, 1, 2, 3], [1, 3, 0, 2, 1, 3]])
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    size = (4, 4)
    input = torch.sparse_coo_tensor(indices, values, size, requires_grad=False).to_dense().numpy()
    mat2 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]).numpy()

    input_dict = {
        "input": input,
        "mat2": mat2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Sparse matrix with negative values
    indices = torch.tensor([[0, 1], [1, 0], [2, 2]])
    values = torch.tensor([-1.0, 2.0, -3.0])
    size = (3, 3)
    input = torch.sparse_coo_tensor(indices, values, size, requires_grad=False).to_dense().numpy()
    mat2 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).numpy()

    input_dict = {
        "input": input,
        "mat2": mat2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Rectangular matrices
    indices = torch.tensor([[0, 1], [1, 0], [2, 1]])
    values = torch.tensor([1.0, 2.0, 3.0])
    size = (3, 4)
    input = torch.sparse_coo_tensor(indices, values, size, requires_grad=False).to_dense().numpy()
    mat2 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]).numpy()

    input_dict = {
        "input": input,
        "mat2": mat2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Sparse matrix with some zero values, should still work
    indices = torch.tensor([[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]])
    values = torch.tensor([1.0, 0.0, 0.0, 4.0, 5.0, 0.0])
    size = (3, 2)
    input = torch.sparse_coo_tensor(indices, values, size, requires_grad=False).to_dense().numpy()
    mat2 = torch.tensor([[1.0], [2.0]]).numpy()

    input_dict = {
        "input": input,
        "mat2": mat2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.spmm"] = spmm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.spmm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.spmm'.")

check_valid('torch.spmm', generated_inputs['torch.spmm'], lib="torch", suffix=0)
