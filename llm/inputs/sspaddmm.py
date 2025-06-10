
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

generated_inputs = dict()

import torch
import copy
import numpy as np

def sspaddmm_inputs():
    list_of_inputs = []

    # Case 1: Basic case with float tensors
    input_sparse = torch.sparse_coo_tensor(torch.randint(0, 5, (2, 3)), torch.randn(3), (5, 4)).to_dense().numpy()
    mat1_sparse = torch.sparse_coo_tensor(torch.randint(0, 5, (2, 3)), torch.randn(3), (5, 3)).to_dense().numpy()
    mat2_dense = torch.randn(3, 4).numpy()
    beta = 1.0
    alpha = 1.0

    input_dict = {
        "input": input_sparse,
        "mat1": mat1_sparse,
        "mat2": mat2_dense,
        "beta": beta,
        "alpha": alpha
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Different beta and alpha values, integer tensors
    input_sparse = torch.sparse_coo_tensor(torch.randint(0, 3, (2, 2)), torch.randint(1, 5, (2,)).float(), (3, 4)).to_dense().numpy()
    mat1_sparse = torch.sparse_coo_tensor(torch.randint(0, 3, (2, 2)), torch.randint(1, 5, (2,)).float(), (3, 2)).to_dense().numpy()
    mat2_dense = torch.randint(1, 5, (2, 4)).float().numpy()
    beta = 0.5
    alpha = 2.0

    input_dict = {
        "input": input_sparse,
        "mat1": mat1_sparse,
        "mat2": mat2_dense,
        "beta": beta,
        "alpha": alpha
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Negative values, different dimensions
    input_sparse = torch.sparse_coo_tensor(torch.randint(0, 3, (2, 2)), torch.randn(2), (3, 4)).to_dense().numpy()
    mat1_sparse = torch.sparse_coo_tensor(torch.randint(0, 3, (2, 2)), torch.randn(2), (3, 2)).to_dense().numpy()
    mat2_dense = torch.randn(2, 4).numpy()
    beta = -1.0
    alpha = 0.5

    input_dict = {
        "input": input_sparse,
        "mat1": mat1_sparse,
        "mat2": mat2_dense,
        "beta": beta,
        "alpha": alpha
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Smaller Matrices
    input_sparse = torch.sparse_coo_tensor(torch.randint(0, 3, (2, 2)), torch.randn(2), (3, 4)).to_dense().numpy()
    mat1_sparse = torch.sparse_coo_tensor(torch.randint(0, 3, (2, 2)), torch.randn(2), (3, 2)).to_dense().numpy()
    mat2_dense = torch.randn(2, 4).numpy()
    beta = 0.8
    alpha = 0.9

    input_dict = {
        "input": input_sparse,
        "mat1": mat1_sparse,
        "mat2": mat2_dense,
        "beta": beta,
        "alpha": alpha
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

if __name__ == '__main__':
    generated_inputs = {}
    generated_inputs["torch.sspaddmm"] = sspaddmm_inputs()
    print(generated_inputs)

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('sspaddmm', generated_inputs['torch.sspaddmm'], lib="torch")
