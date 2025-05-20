
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def sspaddmm_inputs():
    list_of_inputs = []

    # Case 1: Basic case with float tensors and default beta, alpha
    input_indices = np.array([[0, 0], [1, 2], [2, 1]], dtype=np.int64)
    input_values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_shape = (3, 3)
    input_sparse = torch.sparse_coo_tensor(torch.tensor(input_indices).t(), input_values, input_shape).coalesce()

    mat1_indices = np.array([[0, 0], [1, 1], [2, 0]], dtype=np.int64)
    mat1_values = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    mat1_shape = (3, 2)
    mat1_sparse = torch.sparse_coo_tensor(torch.tensor(mat1_indices).t(), mat1_values, mat1_shape).coalesce()
    
    mat2_dense = np.random.rand(2, 4).astype(np.float32)

    input_dict = {
        "input": input_sparse,
        "mat1": mat1_sparse,
        "mat2": mat2_dense,
        "beta": 1.0,
        "alpha": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Integer tensors, negative beta and alpha
    input_indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    input_values = np.array([1, 2], dtype=np.int32)
    input_shape = (2, 3)
    input_sparse = torch.sparse_coo_tensor(torch.tensor(input_indices).t(), input_values, input_shape).coalesce()

    mat1_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    mat1_values = np.array([4, 5], dtype=np.int32)
    mat1_shape = (2, 2)
    mat1_sparse = torch.sparse_coo_tensor(torch.tensor(mat1_indices).t(), mat1_values, mat1_shape).coalesce()
    
    mat2_dense = np.random.randint(-5, 5, size=(2, 5)).astype(np.int32)

    input_dict = {
        "input": input_sparse,
        "mat1": mat1_sparse,
        "mat2": mat2_dense,
        "beta": -0.5,
        "alpha": -2.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Different dimensions and beta=0
    input_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    input_values = np.array([1.0, 2.0], dtype=np.float64)
    input_shape = (2, 3)
    input_sparse = torch.sparse_coo_tensor(torch.tensor(input_indices).t(), input_values, input_shape).coalesce()

    mat1_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    mat1_values = np.array([4.0, 5.0], dtype=np.float64)
    mat1_shape = (2, 2)
    mat1_sparse = torch.sparse_coo_tensor(torch.tensor(mat1_indices).t(), mat1_values, mat1_shape).coalesce()
    
    mat2_dense = np.random.rand(2, 6).astype(np.float64)

    input_dict = {
        "input": input_sparse,
        "mat1": mat1_sparse,
        "mat2": mat2_dense,
        "beta": 0.0,
        "alpha": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Small matrices
    input_indices = np.array([[0, 0]], dtype=np.int64)
    input_values = np.array([1.0], dtype=np.float32)
    input_shape = (1, 1)
    input_sparse = torch.sparse_coo_tensor(torch.tensor(input_indices).t(), input_values, input_shape).coalesce()

    mat1_indices = np.array([[0, 0]], dtype=np.int64)
    mat1_values = np.array([4.0], dtype=np.float32)
    mat1_shape = (1, 1)
    mat1_sparse = torch.sparse_coo_tensor(torch.tensor(mat1_indices).t(), mat1_values, mat1_shape).coalesce()
    
    mat2_dense = np.random.rand(1, 1).astype(np.float32)

    input_dict = {
        "input": input_sparse,
        "mat1": mat1_sparse,
        "mat2": mat2_dense,
        "beta": 1.0,
        "alpha": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Larger matrices
    input_indices = np.array([[0, 0], [1, 2], [2, 1], [5, 4]], dtype=np.int64)
    input_values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    input_shape = (6, 5)
    input_sparse = torch.sparse_coo_tensor(torch.tensor(input_indices).t(), input_values, input_shape).coalesce()

    mat1_indices = np.array([[0, 0], [1, 1], [2, 0], [4, 3]], dtype=np.int64)
    mat1_values = np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32)
    mat1_shape = (6, 4)
    mat1_sparse = torch.sparse_coo_tensor(torch.tensor(mat1_indices).t(), mat1_values, mat1_shape).coalesce()
    
    mat2_dense = np.random.rand(4, 7).astype(np.float32)

    input_dict = {
        "input": input_sparse,
        "mat1": mat1_sparse,
        "mat2": mat2_dense,
        "beta": 0.8,
        "alpha": 1.2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    #Case 6 : 2 Dimensions for input and mat1
    input_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    input_values = np.array([1.0, 2.0], dtype=np.float32)
    input_shape = (2, 2)
    input_sparse = torch.sparse_coo_tensor(torch.tensor(input_indices).t(), input_values, input_shape).coalesce()

    mat1_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    mat1_values = np.array([4.0, 5.0], dtype=np.float32)
    mat1_shape = (2, 2)
    mat1_sparse = torch.sparse_coo_tensor(torch.tensor(mat1_indices).t(), mat1_values, mat1_shape).coalesce()
    
    mat2_dense = np.random.rand(2, 3).astype(np.float32)

    input_dict = {
        "input": input_sparse,
        "mat1": mat1_sparse,
        "mat2": mat2_dense,
        "beta": 0.8,
        "alpha": 1.2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = sspaddmm_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('sspaddmm', generated_inputs)
