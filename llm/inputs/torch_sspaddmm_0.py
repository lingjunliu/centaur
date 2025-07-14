
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def sspaddmm_inputs():
    list_of_inputs = []

    # Input 1: Basic example
    indices = torch.tensor([[0, 1], [1, 0]])
    values = torch.tensor([1.0, 2.0])
    size = torch.Size([2, 2])
    input_sparse = torch.sparse_coo_tensor(indices, values, size).to_dense().numpy()
    mat1_sparse = torch.sparse_coo_tensor(torch.tensor([[0, 0], [1, 1]]), torch.tensor([3.0, 4.0]), (2, 2)).to_dense().numpy()
    mat2_dense = np.array([[5.0, 6.0], [7.0, 8.0]])
    beta = 1.0
    alpha = 1.0
    out = np.array([[0.0, 0.0], [0.0, 0.0]])

    input_dict = {
        "input": input_sparse,
        "mat1": mat1_sparse,
        "mat2": mat2_dense,
        "beta": beta,
        "alpha": alpha,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different beta and alpha
    indices = torch.tensor([[0, 0], [1, 1]])
    values = torch.tensor([1.0, 2.0])
    size = torch.Size([2, 2])
    input_sparse = torch.sparse_coo_tensor(indices, values, size).to_dense().numpy()

    indices = torch.tensor([[0, 1], [1, 0]])
    values = torch.tensor([3.0, 4.0])
    size = torch.Size([2, 2])

    mat1_sparse = torch.sparse_coo_tensor(indices, values, size).to_dense().numpy()
    mat2_dense = np.array([[5.0, 6.0], [7.0, 8.0]])
    beta = 0.5
    alpha = 2.0
    out = np.array([[0.0, 0.0], [0.0, 0.0]])

    input_dict = {
        "input": input_sparse,
        "mat1": mat1_sparse,
        "mat2": mat2_dense,
        "beta": beta,
        "alpha": alpha,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative values
    indices = torch.tensor([[0, 1], [1, 0]])
    values = torch.tensor([-1.0, 2.0])
    size = torch.Size([2, 2])

    input_sparse = torch.sparse_coo_tensor(indices, values, size).to_dense().numpy()

    indices = torch.tensor([[0, 0], [1, 1]])
    values = torch.tensor([3.0, -4.0])
    size = torch.Size([2, 2])

    mat1_sparse = torch.sparse_coo_tensor(indices, values, size).to_dense().numpy()
    mat2_dense = np.array([[-5.0, 6.0], [7.0, -8.0]])
    beta = 1.0
    alpha = 1.0
    out = np.array([[0.0, 0.0], [0.0, 0.0]])

    input_dict = {
        "input": input_sparse,
        "mat1": mat1_sparse,
        "mat2": mat2_dense,
        "beta": beta,
        "alpha": alpha,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different shapes (removed as shapes must align for matrix multiplication)
    input_sparse = np.array([[1.0, 0.0], [0.0, 1.0]])
    mat1_sparse = np.array([[1.0, 0.0], [0.0, 1.0]])
    mat2_dense = np.array([[5.0, 6.0], [7.0, 8.0]])
    beta = 1.0
    alpha = 1.0
    out = np.array([[0.0, 0.0], [0.0, 0.0]])

    input_dict = {
        "input": input_sparse,
        "mat1": mat1_sparse,
        "mat2": mat2_dense,
        "beta": beta,
        "alpha": alpha,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger matrices
    indices = torch.tensor([[0, 0], [1, 1], [2, 2]])
    values = torch.tensor([1.0, 2.0, 3.0])
    size = torch.Size([3, 3])
    input_sparse = torch.sparse_coo_tensor(indices, values, size).to_dense().numpy()
    
    indices = torch.tensor([[0, 1], [1, 2], [2, 0]])
    values = torch.tensor([3.0, 4.0, 5.0])
    size = torch.Size([3, 3])
    mat1_sparse = torch.sparse_coo_tensor(indices, values, size).to_dense().numpy()
    mat2_dense = np.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0], [11.0, 12.0, 13.0]])
    beta = 0.0
    alpha = 1.0
    out = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    input_dict = {
        "input": input_sparse,
        "mat1": mat1_sparse,
        "mat2": mat2_dense,
        "beta": beta,
        "alpha": alpha,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: beta = 0
    indices = torch.tensor([[0, 0], [1, 1]])
    values = torch.tensor([1.0, 2.0])
    size = torch.Size([2, 2])

    input_sparse = torch.sparse_coo_tensor(indices, values, size).to_dense().numpy()

    indices = torch.tensor([[0, 1], [1, 0]])
    values = torch.tensor([3.0, 4.0])
    size = torch.Size([2, 2])

    mat1_sparse = torch.sparse_coo_tensor(indices, values, size).to_dense().numpy()
    mat2_dense = np.array([[5.0, 6.0], [7.0, 8.0]])
    beta = 0.0
    alpha = 1.0
    out = np.array([[0.0, 0.0], [0.0, 0.0]])

    input_dict = {
        "input": input_sparse,
        "mat1": mat1_sparse,
        "mat2": mat2_dense,
        "beta": beta,
        "alpha": alpha,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: alpha = 0
    indices = torch.tensor([[0, 0], [1, 1]])
    values = torch.tensor([1.0, 2.0])
    size = torch.Size([2, 2])

    input_sparse = torch.sparse_coo_tensor(indices, values, size).to_dense().numpy()

    indices = torch.tensor([[0, 1], [1, 0]])
    values = torch.tensor([3.0, 4.0])
    size = torch.Size([2, 2])

    mat1_sparse = torch.sparse_coo_tensor(indices, values, size).to_dense().numpy()
    mat2_dense = np.array([[5.0, 6.0], [7.0, 8.0]])
    beta = 1.0
    alpha = 0.0
    out = np.array([[0.0, 0.0], [0.0, 0.0]])

    input_dict = {
        "input": input_sparse,
        "mat1": mat1_sparse,
        "mat2": mat2_dense,
        "beta": beta,
        "alpha": alpha,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Identity matrix
    indices = torch.tensor([[0, 0], [1, 1]])
    values = torch.tensor([1.0, 1.0])
    size = torch.Size([2, 2])

    input_sparse = torch.sparse_coo_tensor(indices, values, size).to_dense().numpy()

    indices = torch.tensor([[0, 0], [1, 1]])
    values = torch.tensor([1.0, 1.0])
    size = torch.Size([2, 2])

    mat1_sparse = torch.sparse_coo_tensor(indices, values, size).to_dense().numpy()
    mat2_dense = np.array([[1.0, 0.0], [0.0, 1.0]])
    beta = 1.0
    alpha = 1.0
    out = np.array([[0.0, 0.0], [0.0, 0.0]])

    input_dict = {
        "input": input_sparse,
        "mat1": mat1_sparse,
        "mat2": mat2_dense,
        "beta": beta,
        "alpha": alpha,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different values
    indices = torch.tensor([[0, 0], [1, 1]])
    values = torch.tensor([2.5, 3.5])
    size = torch.Size([2, 2])

    input_sparse = torch.sparse_coo_tensor(indices, values, size).to_dense().numpy()

    indices = torch.tensor([[0, 1], [1, 0]])
    values = torch.tensor([1.5, 2.5])
    size = torch.Size([2, 2])

    mat1_sparse = torch.sparse_coo_tensor(indices, values, size).to_dense().numpy()
    mat2_dense = np.array([[3.5, 4.5], [5.5, 6.5]])
    beta = 0.75
    alpha = 1.25
    out = np.array([[0.0, 0.0], [0.0, 0.0]])

    input_dict = {
        "input": input_sparse,
        "mat1": mat1_sparse,
        "mat2": mat2_dense,
        "beta": beta,
        "alpha": alpha,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: All zeros
    indices = torch.tensor([[0, 0], [1, 1]])
    values = torch.tensor([0.0, 0.0])
    size = torch.Size([2, 2])
    input_sparse = torch.sparse_coo_tensor(indices, values, size).to_dense().numpy()
    
    indices = torch.tensor([[0, 0], [1, 1]])
    values = torch.tensor([0.0, 0.0])
    size = torch.Size([2, 2])

    mat1_sparse = torch.sparse_coo_tensor(indices, values, size).to_dense().numpy()
    mat2_dense = np.array([[0.0, 0.0], [0.0, 0.0]])
    beta = 1.0
    alpha = 1.0
    out = np.array([[0.0, 0.0], [0.0, 0.0]])

    input_dict = {
        "input": input_sparse,
        "mat1": mat1_sparse,
        "mat2": mat2_dense,
        "beta": beta,
        "alpha": alpha,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.sspaddmm"] = sspaddmm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sspaddmm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sspaddmm'.")

check_valid('torch.sspaddmm', generated_inputs['torch.sspaddmm'], lib="torch", suffix=0)
