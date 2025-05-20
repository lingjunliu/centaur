
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def spmm_inputs():
    list_of_inputs = []

    # Input 1: Basic float sparse matrix and dense matrix
    indices = torch.tensor([[0, 1], [1, 2], [2, 0]]).t()
    values = torch.tensor([1.0, 2.0, 3.0])
    shape = (3, 3)
    sparse_matrix = torch.sparse_coo_tensor(
        indices,
        values,
        shape
    ).coalesce().numpy()
    dense_matrix = torch.randn(3, 4).numpy()
    input_dict = {"input": sparse_matrix, "mat2": dense_matrix}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float sparse matrix and dense matrix with different shapes
    indices = torch.tensor([[0, 0], [1, 1], [2, 2]]).t()
    values = torch.tensor([1.0, 1.0, 1.0])
    shape = (3, 3)
    sparse_matrix = torch.sparse_coo_tensor(
        indices,
        values,
        shape
    ).coalesce().numpy()
    dense_matrix = torch.randn(3, 2).numpy()
    input_dict = {"input": sparse_matrix, "mat2": dense_matrix}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Int sparse matrix and dense matrix
    indices = torch.tensor([[0, 1], [1, 2], [2, 0]]).t()
    values = torch.tensor([1, 2, 3])
    shape = (3, 3)
    sparse_matrix = torch.sparse_coo_tensor(
        indices,
        values,
        shape
    ).coalesce().numpy()
    dense_matrix = torch.randint(0, 5, (3, 4)).numpy()
    input_dict = {"input": sparse_matrix, "mat2": dense_matrix}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Float sparse matrix with negative values and dense matrix
    indices = torch.tensor([[0, 1], [1, 2], [2, 0]]).t()
    values = torch.tensor([-1.0, 2.0, -3.0])
    shape = (3, 3)
    sparse_matrix = torch.sparse_coo_tensor(
        indices,
        values,
        shape
    ).coalesce().numpy()
    dense_matrix = torch.randn(3, 4).numpy()
    input_dict = {"input": sparse_matrix, "mat2": dense_matrix}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger float sparse matrix and dense matrix
    indices = torch.tensor([[0, 1], [1, 2], [2, 0], [3, 1], [4, 3]]).t()
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    shape = (5, 5)
    sparse_matrix = torch.sparse_coo_tensor(
        indices,
        values,
        shape
    ).coalesce().numpy()
    dense_matrix = torch.randn(5, 6).numpy()
    input_dict = {"input": sparse_matrix, "mat2": dense_matrix}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Rectangular sparse matrix
    indices = torch.tensor([[0, 1], [1, 2]]).t()
    values = torch.tensor([1.0, 2.0])
    shape = (2, 3)
    sparse_matrix = torch.sparse_coo_tensor(
        indices,
        values,
        shape
    ).coalesce().numpy()
    dense_matrix = torch.randn(3, 4).numpy()
    input_dict = {"input": sparse_matrix, "mat2": dense_matrix}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = spmm_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('spmm', generated_inputs)
