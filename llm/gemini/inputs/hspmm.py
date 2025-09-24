
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def hspmm_inputs():
    list_of_inputs = []

    rows = 10
    cols = 5
    mat1_indices = torch.randint(0, 2, (2, 15))
    mat1_values = torch.randn(15)
    mat1 = torch.sparse_coo_tensor(mat1_indices, mat1_values, (rows, cols))
    mat2 = torch.randn(cols, 8)

    input_dict = {
        "mat1": mat1.to_dense().numpy(),
        "mat2": mat2.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    rows = 5
    cols = 10
    mat1_indices = torch.randint(0, 2, (2, 20))
    mat1_values = torch.randn(20)
    mat1 = torch.sparse_coo_tensor(mat1_indices, mat1_values, (rows, cols))
    mat2 = torch.randn(cols, 12)

    input_dict = {
        "mat1": mat1.to_dense().numpy(),
        "mat2": mat2.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    rows = 8
    cols = 3
    mat1_indices = torch.randint(0, 2, (2, 10))
    mat1_values = torch.randn(10)
    mat1 = torch.sparse_coo_tensor(mat1_indices, mat1_values, (rows, cols))
    mat2 = torch.randn(cols, 5)

    input_dict = {
        "mat1": mat1.to_dense().numpy(),
        "mat2": mat2.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    rows = 4
    cols = 7
    mat1_indices = torch.randint(0, 2, (2, 8))
    mat1_values = torch.randn(8)
    mat1 = torch.sparse_coo_tensor(mat1_indices, mat1_values, (rows, cols))
    mat2 = torch.randn(cols, 2)

    input_dict = {
        "mat1": mat1.to_dense().numpy(),
        "mat2": mat2.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    rows = 6
    cols = 9
    mat1_indices = torch.randint(0, 2, (2, 18))
    mat1_values = torch.randn(18)
    mat1 = torch.sparse_coo_tensor(mat1_indices, mat1_values, (rows, cols))
    mat2 = torch.randn(cols, 10)
    
    input_dict = {
        "mat1": mat1.to_dense().numpy(),
        "mat2": mat2.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = hspmm_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('hspmm', generated_inputs)
