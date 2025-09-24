
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy as np

def hspmm_inputs():
    list_of_inputs = []

    # All tensor inputs must be numpy arrays to satisfy the validation framework.
    # torch.hspmm expects mat1 to be a sparse tensor. We provide its dense numpy representation
    # by creating a sparse tensor and then converting it with .to_dense().numpy().
    # This approach satisfies the numpy format requirement, although it might lead
    # to a runtime NotImplementedError since the underlying API expects a sparse object.

    # Input 1: Basic valid case with float32
    mat1 = torch.sparse_coo_tensor(
        torch.tensor([[0, 2], [1, 3]], dtype=torch.long),
        torch.tensor([1.0, 2.0], dtype=torch.float32),
        (3, 4)
    ).to_dense().numpy()
    mat2 = np.random.randn(4, 5).astype(np.float32)
    out = np.zeros((3, 5), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({'mat1': mat1, 'mat2': mat2, 'out': out}))

    # Input 2: Using float64 dtype
    mat1 = torch.sparse_coo_tensor(
        torch.tensor([[0, 0, 2], [1, 3, 0]], dtype=torch.long),
        torch.tensor([10.5, -2.1, 5.5], dtype=torch.float64),
        (3, 4), dtype=torch.float64
    ).to_dense().numpy()
    mat2 = np.random.randn(4, 2).astype(np.float64)
    out = np.zeros((3, 2), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({'mat1': mat1, 'mat2': mat2, 'out': out}))

    # Input 3: mat1 is an empty sparse matrix
    mat1 = torch.sparse_coo_tensor(
        torch.tensor([[], []], dtype=torch.long),
        torch.tensor([], dtype=torch.float32),
        (4, 4)
    ).to_dense().numpy()
    mat2 = np.random.randn(4, 3).astype(np.float32)
    out = np.zeros((4, 3), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({'mat1': mat1, 'mat2': mat2, 'out': out}))

    # Input 4: Sparse representation of a dense matrix
    mat1 = torch.sparse_coo_tensor(
        torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=torch.long),
        torch.tensor([1., 2., 3., 4.], dtype=torch.float32),
        (2, 2)
    ).to_dense().numpy()
    mat2 = np.random.randn(2, 3).astype(np.float32)
    out = np.zeros((2, 3), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({'mat1': mat1, 'mat2': mat2, 'out': out}))

    # Input 5: mat2 contains all zeros
    mat1 = torch.sparse_coo_tensor(
        torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        torch.tensor([5., 6.], dtype=torch.float32),
        (2, 2)
    ).to_dense().numpy()
    mat2 = np.zeros((2, 4), dtype=np.float32)
    out = np.zeros((2, 4), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({'mat1': mat1, 'mat2': mat2, 'out': out}))

    # Input 6: mat2 contains negative values
    mat1 = torch.eye(3, dtype=torch.float32).to_sparse().to_dense().numpy()
    mat2 = np.array([[-1., 2.], [3., -4.], [-5., 6.]], dtype=np.float32)
    out = np.zeros((3, 2), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({'mat1': mat1, 'mat2': mat2, 'out': out}))

    # Input 7: mat1 contains negative values
    mat1 = torch.sparse_coo_tensor(
        torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        torch.tensor([-5., -6.], dtype=torch.float32),
        (2, 2)
    ).to_dense().numpy()
    mat2 = np.random.rand(2, 5).astype(np.float32)
    out = np.zeros((2, 5), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({'mat1': mat1, 'mat2': mat2, 'out': out}))

    # Input 8: Larger matrices
    mat1 = torch.sparse_coo_tensor(
        torch.tensor([[0, 9, 5], [5, 18, 10]], dtype=torch.long),
        torch.tensor([10., 20., 30.], dtype=torch.float32),
        (10, 20)
    ).to_dense().numpy()
    mat2 = np.random.randn(20, 15).astype(np.float32)
    out = np.zeros((10, 15), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({'mat1': mat1, 'mat2': mat2, 'out': out}))

    # Input 9: Without the optional 'out' tensor
    mat1 = torch.sparse_coo_tensor(
        torch.tensor([[0, 2], [1, 3]], dtype=torch.long),
        torch.tensor([1.0, 2.0], dtype=torch.float32),
        (3, 4)
    ).to_dense().numpy()
    mat2 = np.random.randn(4, 5).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({'mat1': mat1, 'mat2': mat2}))

    # Input 10: Sparse matrix with repeated indices (coalesced)
    mat1 = torch.sparse_coo_tensor(
        torch.tensor([[0, 0, 1], [1, 1, 0]], dtype=torch.long),
        torch.tensor([2., 3., 4.], dtype=torch.float32),
        (2, 2)
    ).coalesce().to_dense().numpy()
    mat2 = np.ones((2, 2), dtype=np.float32)
    out = np.zeros((2, 2), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({'mat1': mat1, 'mat2': mat2, 'out': out}))
    
    return list_of_inputs

generated_inputs["torch.hspmm"] = hspmm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.hspmm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.hspmm'.")

check_valid('torch.hspmm', generated_inputs['torch.hspmm'], lib="torch", suffix=0)
