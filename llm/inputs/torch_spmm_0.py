
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def spmm_inputs():
    list_of_inputs = []

    # Input 1: Basic example
    sparse_matrix = torch.sparse_coo_tensor(indices=torch.tensor([[0, 1], [1, 0]]), values=torch.tensor([1.0, 2.0]), size=(2, 2)).to_dense()
    dense_matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    input_dict = {"input": sparse_matrix.numpy(), "mat2": dense_matrix}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Larger matrices
    sparse_matrix = torch.sparse_coo_tensor(indices=torch.tensor([[0, 1, 2], [1, 2, 0]]), values=torch.tensor([1.0, 2.0, 3.0]), size=(3, 3)).to_dense()
    dense_matrix = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).numpy()
    input_dict = {"input": sparse_matrix.numpy(), "mat2": dense_matrix}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Rectangular matrices
    sparse_matrix = torch.sparse_coo_tensor(indices=torch.tensor([[0, 1], [1, 0]]), values=torch.tensor([1.0, 2.0]), size=(2, 3)).to_dense()
    dense_matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).numpy()
    input_dict = {"input": sparse_matrix.numpy(), "mat2": dense_matrix}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different data type
    sparse_matrix = torch.sparse_coo_tensor(indices=torch.tensor([[0, 1], [1, 0]]), values=torch.tensor([1, 2], dtype=torch.int64), size=(2, 2)).to_dense()
    dense_matrix = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64).numpy()
    input_dict = {"input": sparse_matrix.numpy(), "mat2": dense_matrix}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Sparse matrix with some zero values
    sparse_matrix = torch.sparse_coo_tensor(indices=torch.tensor([[0, 1], [1, 0]]), values=torch.tensor([0.0, 2.0]), size=(2, 2)).to_dense()
    dense_matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    input_dict = {"input": sparse_matrix.numpy(), "mat2": dense_matrix}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty sparse matrix
    sparse_matrix = torch.sparse_coo_tensor(indices=torch.empty((2,0)), values=torch.empty((0,)), size=(2, 2)).to_dense()
    dense_matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    input_dict = {"input": sparse_matrix.numpy(), "mat2": dense_matrix}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Single element sparse matrix
    sparse_matrix = torch.sparse_coo_tensor(indices=torch.tensor([[0], [0]]), values=torch.tensor([1.0]), size=(1, 1)).to_dense()
    dense_matrix = torch.tensor([[2.0]]).numpy()
    input_dict = {"input": sparse_matrix.numpy(), "mat2": dense_matrix}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative values in sparse matrix
    sparse_matrix = torch.sparse_coo_tensor(indices=torch.tensor([[0, 1], [1, 0]]), values=torch.tensor([-1.0, 2.0]), size=(2, 2)).to_dense()
    dense_matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    input_dict = {"input": sparse_matrix.numpy(), "mat2": dense_matrix}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative values in dense matrix
    sparse_matrix = torch.sparse_coo_tensor(indices=torch.tensor([[0, 1], [1, 0]]), values=torch.tensor([1.0, 2.0]), size=(2, 2)).to_dense()
    dense_matrix = torch.tensor([[-1.0, 2.0], [3.0, -4.0]]).numpy()
    input_dict = {"input": sparse_matrix.numpy(), "mat2": dense_matrix}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Sparse matrix with duplicate coordinates
    sparse_matrix = torch.sparse_coo_tensor(indices=torch.tensor([[0, 0, 1], [0, 0, 1]]), values=torch.tensor([1.0, 2.0, 3.0]), size=(2, 2)).to_dense()
    dense_matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    input_dict = {"input": sparse_matrix.numpy(), "mat2": dense_matrix}
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
