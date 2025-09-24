
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy
import scipy.sparse

def sspaddmm_inputs():
    list_of_inputs = []

    def _create_scipy_sparse_matrix(shape, density=0.2, dtype=np.float32):
        # Using scipy.sparse to create a sparse matrix representation.
        # This is a common way to handle sparse data and its .dtype is a numpy dtype,
        # which should be compatible with the testing framework.
        return scipy.sparse.random(shape[0], shape[1], density=density, format='coo', dtype=dtype)

    # Case 1: Basic valid case (float32)
    m, k, n = 4, 5, 6
    input_tensor = _create_scipy_sparse_matrix((m, n), dtype=np.float32)
    mat1 = _create_scipy_sparse_matrix((m, k), dtype=np.float32)
    mat2 = np.random.randn(k, n).astype(np.float32)
    out_tensor = _create_scipy_sparse_matrix((m, n), density=0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({
        'input': input_tensor,
        'mat1': mat1,
        'mat2': mat2,
        'beta': 1.0,
        'alpha': 1.0,
        'out': out_tensor
    }))

    # Case 2: Out tensor with pre-existing values
    m, k, n = 3, 3, 3
    input_tensor = _create_scipy_sparse_matrix((m, n))
    mat1 = _create_scipy_sparse_matrix((m, k))
    mat2 = np.random.randn(k, n).astype(np.float32)
    out_tensor = _create_scipy_sparse_matrix((m, n), density=0.5)
    list_of_inputs.append(copy.deepcopy({
        'input': input_tensor,
        'mat1': mat1,
        'mat2': mat2,
        'beta': 1.0,
        'alpha': 1.0,
        'out': out_tensor
    }))

    # Case 3: float64 dtype
    m, k, n = 5, 2, 4
    input_tensor = _create_scipy_sparse_matrix((m, n), dtype=np.float64)
    mat1 = _create_scipy_sparse_matrix((m, k), dtype=np.float64)
    mat2 = np.random.randn(k, n).astype(np.float64)
    out_tensor = _create_scipy_sparse_matrix((m, n), density=0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({
        'input': input_tensor,
        'mat1': mat1,
        'mat2': mat2,
        'beta': 1.0,
        'alpha': 1.0,
        'out': out_tensor
    }))

    # Case 4: Custom `beta` and `alpha`
    m, k, n = 4, 4, 4
    input_tensor = _create_scipy_sparse_matrix((m, n))
    mat1 = _create_scipy_sparse_matrix((m, k))
    mat2 = np.random.randn(k, n).astype(np.float32)
    out_tensor = _create_scipy_sparse_matrix((m, n), density=0)
    list_of_inputs.append(copy.deepcopy({
        'input': input_tensor,
        'mat1': mat1,
        'mat2': mat2,
        'beta': 2.5,
        'alpha': -1.5,
        'out': out_tensor
    }))

    # Case 5: `beta = 0.0`
    m, k, n = 2, 3, 2
    input_tensor = _create_scipy_sparse_matrix((m, n))
    mat1 = _create_scipy_sparse_matrix((m, k), density=0.8)
    mat2 = np.random.randn(k, n).astype(np.float32)
    out_tensor = _create_scipy_sparse_matrix((m, n))
    list_of_inputs.append(copy.deepcopy({
        'input': input_tensor,
        'mat1': mat1,
        'mat2': mat2,
        'beta': 0.0,
        'alpha': 2.0,
        'out': out_tensor
    }))

    # Case 6: `alpha = 0.0`
    m, k, n = 3, 2, 4
    input_tensor = _create_scipy_sparse_matrix((m, n), density=0.7)
    mat1 = _create_scipy_sparse_matrix((m, k))
    mat2 = np.random.randn(k, n).astype(np.float32)
    out_tensor = _create_scipy_sparse_matrix((m, n), density=0)
    list_of_inputs.append(copy.deepcopy({
        'input': input_tensor,
        'mat1': mat1,
        'mat2': mat2,
        'beta': -1.0,
        'alpha': 0.0,
        'out': out_tensor
    }))

    # Case 7: Negative values in tensors
    m, k, n = 3, 3, 3
    input_tensor = _create_scipy_sparse_matrix((m, n), density=0.6)
    input_tensor.data = -np.abs(input_tensor.data)
    mat1 = _create_scipy_sparse_matrix((m, k), density=0.6)
    mat1.data = -np.abs(mat1.data)
    mat2 = (np.random.randn(k, n) - 2).astype(np.float32)
    out_tensor = _create_scipy_sparse_matrix((m, n), density=0)
    list_of_inputs.append(copy.deepcopy({
        'input': input_tensor,
        'mat1': mat1,
        'mat2': mat2,
        'beta': 0.5,
        'alpha': 0.5,
        'out': out_tensor
    }))

    # Case 8: Larger dimensions with high sparsity
    m, k, n = 20, 30, 25
    input_tensor = _create_scipy_sparse_matrix((m, n), density=0.05)
    mat1 = _create_scipy_sparse_matrix((m, k), density=0.05)
    mat2 = np.random.randn(k, n).astype(np.float32)
    out_tensor = _create_scipy_sparse_matrix((m, n), density=0)
    list_of_inputs.append(copy.deepcopy({
        'input': input_tensor,
        'mat1': mat1,
        'mat2': mat2,
        'beta': 1.0,
        'alpha': 1.0,
        'out': out_tensor
    }))

    # Case 9: Minimal dimensions (1x1 matrices)
    m, k, n = 1, 1, 1
    input_tensor = _create_scipy_sparse_matrix((m, n), density=1.0)
    mat1 = _create_scipy_sparse_matrix((m, k), density=1.0)
    mat2 = np.random.randn(k, n).astype(np.float32)
    out_tensor = _create_scipy_sparse_matrix((m, n), density=0)
    list_of_inputs.append(copy.deepcopy({
        'input': input_tensor,
        'mat1': mat1,
        'mat2': mat2,
        'beta': 1.0,
        'alpha': 1.0,
        'out': out_tensor
    }))

    # Case 10: Non-square matrices and different beta/alpha
    m, k, n = 6, 2, 8
    input_tensor = _create_scipy_sparse_matrix((m, n), density=0.3)
    mat1 = _create_scipy_sparse_matrix((m, k), density=0.5)
    mat2 = np.random.randn(k, n).astype(np.float32)
    out_tensor = _create_scipy_sparse_matrix((m, n), density=0)
    list_of_inputs.append(copy.deepcopy({
        'input': input_tensor,
        'mat1': mat1,
        'mat2': mat2,
        'beta': 3.0,
        'alpha': -2.0,
        'out': out_tensor
    }))

    return list_of_inputs

generated_inputs["torch.sspaddmm"] = sspaddmm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.sspaddmm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sspaddmm'.")

check_valid('torch.sspaddmm', generated_inputs['torch.sspaddmm'], lib="torch", suffix=0)
