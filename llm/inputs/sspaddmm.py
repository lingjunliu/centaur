
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import scipy.sparse as sparse
import numpy as np
import copy

def sspaddmm_inputs():
    list_of_inputs = []

    # Test case 1: Basic case with float tensors
    input_sparse = sparse.random(5, 5, density=0.5, format="coo", dtype=np.float32)
    input_tensor = torch.sparse_coo_tensor(torch.tensor([input_sparse.row.astype(np.int64), input_sparse.col.astype(np.int64)]), torch.from_numpy(input_sparse.data), size=input_sparse.shape).to_dense().numpy()
    mat1_sparse = sparse.random(5, 3, density=0.5, format="coo", dtype=np.float32)
    mat1_tensor = torch.sparse_coo_tensor(torch.tensor([mat1_sparse.row.astype(np.int64), mat1_sparse.col.astype(np.int64)]), torch.from_numpy(mat1_sparse.data), size=mat1_sparse.shape).to_dense().numpy()
    mat2 = torch.randn(3, 4).numpy()
    input_dict = {"input": input_tensor, "mat1": mat1_tensor, "mat2": mat2, "beta": 1.0, "alpha": 1.0}
    
    try:
        torch.sspaddmm(torch.from_numpy(input_tensor), torch.from_numpy(mat1_tensor), torch.from_numpy(mat2), beta=1.0, alpha=1.0)
        list_of_inputs.append(copy.deepcopy(input_dict))
    except RuntimeError:
        pass

    # Test case 2: Different shapes and beta/alpha values
    input_sparse = sparse.random(3, 4, density=0.3, format="coo", dtype=np.float64)
    input_tensor = torch.sparse_coo_tensor(torch.tensor([input_sparse.row.astype(np.int64), input_sparse.col.astype(np.int64)]), torch.from_numpy(input_sparse.data), size=input_sparse.shape).to_dense().numpy()
    mat1_sparse = sparse.random(3, 2, density=0.3, format="coo", dtype=np.float64)
    mat1_tensor = torch.sparse_coo_tensor(torch.tensor([mat1_sparse.row.astype(np.int64), mat1_sparse.col.astype(np.int64)]), torch.from_numpy(mat1_sparse.data), size=mat1_sparse.shape).to_dense().numpy()
    mat2 = torch.randn(2, 4).numpy()
    input_dict = {"input": input_tensor, "mat1": mat1_tensor, "mat2": mat2, "beta": 0.5, "alpha": 2.0}
    
    try:
        torch.sspaddmm(torch.from_numpy(input_tensor), torch.from_numpy(mat1_tensor), torch.from_numpy(mat2), beta=0.5, alpha=2.0)
        list_of_inputs.append(copy.deepcopy(input_dict))
    except RuntimeError:
        pass

    # Test case 3: Integer tensors
    input_sparse = sparse.random(4, 4, density=0.4, format="coo", dtype=np.int32)
    input_tensor = torch.sparse_coo_tensor(torch.tensor([input_sparse.row.astype(np.int64), input_sparse.col.astype(np.int64)]), torch.from_numpy(input_sparse.data.astype(np.float32)), size=input_sparse.shape).to_dense().numpy()
    mat1_sparse = sparse.random(4, 2, density=0.4, format="coo", dtype=np.int32)
    mat1_tensor = torch.sparse_coo_tensor(torch.tensor([mat1_sparse.row.astype(np.int64), mat1_sparse.col.astype(np.int64)]), torch.from_numpy(mat1_sparse.data.astype(np.float32)), size=mat1_sparse.shape).to_dense().numpy()
    mat2 = torch.randint(0, 10, (2, 3)).numpy()
    input_dict = {"input": input_tensor, "mat1": mat1_tensor, "mat2": mat2, "beta": 0.8, "alpha": 1.5}
    
    try:
        torch.sspaddmm(torch.from_numpy(input_tensor), torch.from_numpy(mat1_tensor), torch.from_numpy(mat2), beta=0.8, alpha=1.5)
        list_of_inputs.append(copy.deepcopy(input_dict))
    except RuntimeError:
        pass

    # Test case 4: Negative values and different sparsity
    input_sparse = sparse.random(2, 5, density=0.1, format="coo", dtype=np.float32)
    input_tensor = torch.sparse_coo_tensor(torch.tensor([input_sparse.row.astype(np.int64), input_sparse.col.astype(np.int64)]), torch.from_numpy(input_sparse.data), size=input_sparse.shape).to_dense().numpy()
    mat1_sparse = sparse.random(2, 3, density=0.1, format="coo", dtype=np.float32)
    mat1_tensor = torch.sparse_coo_tensor(torch.tensor([mat1_sparse.row.astype(np.int64), mat1_sparse.col.astype(np.int64)]), torch.from_numpy(mat1_sparse.data), size=mat1_sparse.shape).to_dense().numpy()
    mat2 = torch.randn(3, 5).numpy() * -1
    input_dict = {"input": input_tensor, "mat1": mat1_tensor, "mat2": mat2, "beta": -1.0, "alpha": 0.5}
    
    try:
        torch.sspaddmm(torch.from_numpy(input_tensor), torch.from_numpy(mat1_tensor), torch.from_numpy(mat2), beta=-1.0, alpha=0.5)
        list_of_inputs.append(copy.deepcopy(input_dict))
    except RuntimeError:
        pass

    # Test case 5: Small matrices
    input_sparse = sparse.random(1, 1, density=1.0, format="coo", dtype=np.float64)
    input_tensor = torch.sparse_coo_tensor(torch.tensor([input_sparse.row.astype(np.int64), input_sparse.col.astype(np.int64)]), torch.from_numpy(input_sparse.data), size=input_sparse.shape).to_dense().numpy()
    mat1_sparse = sparse.random(1, 1, density=1.0, format="coo", dtype=np.float64)
    mat1_tensor = torch.sparse_coo_tensor(torch.tensor([mat1_sparse.row.astype(np.int64), mat1_sparse.col.astype(np.int64)]), torch.from_numpy(mat1_sparse.data), size=mat1_sparse.shape).to_dense().numpy()
    mat2 = torch.randn(1, 1).numpy()
    input_dict = {"input": input_tensor, "mat1": mat1_tensor, "mat2": mat2, "beta": 0.2, "alpha": 0.8}
    
    try:
        torch.sspaddmm(torch.from_numpy(input_tensor), torch.from_numpy(mat1_tensor), torch.from_numpy(mat2), beta=0.2, alpha=0.8)
        list_of_inputs.append(copy.deepcopy(input_dict))
    except RuntimeError:
        pass
    
    # Test case 6: Larger Matrices - Adjusted for potential dimension issues
    input_sparse = sparse.random(3, 5, density=0.2, format="coo", dtype=np.float32)  # Changed to 3x5
    input_tensor = torch.sparse_coo_tensor(torch.tensor([input_sparse.row.astype(np.int64), input_sparse.col.astype(np.int64)]), torch.from_numpy(input_sparse.data), size=input_sparse.shape).to_dense().numpy()
    mat1_sparse = sparse.random(3, 2, density=0.2, format="coo", dtype=np.float32)  # Changed to 3x2
    mat1_tensor = torch.sparse_coo_tensor(torch.tensor([mat1_sparse.row.astype(np.int64), mat1_sparse.col.astype(np.int64)]), torch.from_numpy(mat1_sparse.data), size=mat1_sparse.shape).to_dense().numpy()
    mat2 = torch.randn(2, 5).numpy() # Changed to 2x5
    input_dict = {"input": input_tensor, "mat1": mat1_tensor, "mat2": mat2, "beta": 1.0, "alpha": 1.0}

    try:
        torch.sspaddmm(torch.from_numpy(input_tensor), torch.from_numpy(mat1_tensor), torch.from_numpy(mat2), beta=1.0, alpha=1.0)
        list_of_inputs.append(copy.deepcopy(input_dict))
    except RuntimeError:
        pass

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
