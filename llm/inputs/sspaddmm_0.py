
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def sspaddmm_inputs():
    list_of_inputs = []

    # Case 1: Basic case with float tensors
    indices1 = torch.tensor([[0, 1], [1, 2]]).long()
    values1 = torch.tensor([1.0, 2.0])
    mat1 = torch.sparse_coo_tensor(indices1, values1, (3, 3)).to_dense().numpy()

    mat2 = torch.randn(3, 4).numpy()
    indices_input = torch.tensor([[0, 0], [1, 1]]).long()
    values_input = torch.tensor([3.0, 4.0])
    input = torch.sparse_coo_tensor(indices_input, values_input, (3, 4)).to_dense().numpy()

    input_dict = {
        "input": input,
        "mat1": mat1,
        "mat2": mat2,
        "beta": 1.0,
        "alpha": 1.0,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Different shapes and alpha/beta values
    indices1 = torch.tensor([[0, 0], [1, 1], [2, 2]]).long()
    values1 = torch.tensor([0.5, 1.5, 2.5])
    mat1 = torch.sparse_coo_tensor(indices1, values1, (3, 3)).to_dense().numpy()

    mat2 = torch.randn(3, 2).numpy()
    indices_input = torch.tensor([[0, 0], [1, 1]]).long()
    values_input = torch.tensor([0.7, -0.3])
    input = torch.sparse_coo_tensor(indices_input, values_input, (3, 2)).to_dense().numpy()

    input_dict = {
        "input": input,
        "mat1": mat1,
        "mat2": mat2,
        "beta": 0.5,
        "alpha": 2.0,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Integer tensors
    indices1 = torch.tensor([[0, 1], [1, 0]]).long()
    values1 = torch.tensor([1, 2])
    mat1 = torch.sparse_coo_tensor(indices1, values1, (2, 2)).to_dense().numpy()

    mat2 = torch.randint(0, 5, (2, 3)).numpy()
    indices_input = torch.tensor([[0, 0], [1, 1]]).long()
    values_input = torch.tensor([3, 4])
    input = torch.sparse_coo_tensor(indices_input, values_input, (2, 3)).to_dense().numpy()

    input_dict = {
        "input": input,
        "mat1": mat1,
        "mat2": mat2,
        "beta": 0.0,
        "alpha": 1.0,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Negative values in tensors
    indices1 = torch.tensor([[0, 0], [1, 1]]).long()
    values1 = torch.tensor([-1.0, -2.0])
    mat1 = torch.sparse_coo_tensor(indices1, values1, (2, 2)).to_dense().numpy()

    mat2 = torch.randn(2, 2).numpy()
    indices_input = torch.tensor([[0, 1], [1, 0]]).long()
    values_input = torch.tensor([-3.0, -4.0])
    input = torch.sparse_coo_tensor(indices_input, values_input, (2, 2)).to_dense().numpy()

    input_dict = {
        "input": input,
        "mat1": mat1,
        "mat2": mat2,
        "beta": 1.0,
        "alpha": 1.0,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Different sparse layout and beta=0
    indices1 = torch.tensor([[0, 2], [2, 0]]).long()
    values1 = torch.tensor([1.0, 2.0])
    mat1 = torch.sparse_coo_tensor(indices1, values1, (3, 3)).to_dense().numpy()

    mat2 = torch.randn(3, 4).numpy()
    indices_input = torch.tensor([[1, 1], [2, 3]]).long()
    values_input = torch.tensor([3.0, 4.0])
    input = torch.sparse_coo_tensor(indices_input, values_input, (3, 4)).to_dense().numpy()

    input_dict = {
        "input": input,
        "mat1": mat1,
        "mat2": mat2,
        "beta": 0.0,
        "alpha": 1.0,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 6: Reduce size to prevent OOM and correct sparse_dim
    indices1 = torch.tensor([[0, 0], [1, 1]]).long()
    values1 = torch.tensor([1.0, 2.0])
    mat1 = torch.sparse_coo_tensor(indices1, values1, (2, 2)).to_dense().numpy()

    mat2 = torch.randn(2, 2).numpy()
    indices_input = torch.tensor([[0, 0], [1, 1]]).long()
    values_input = torch.tensor([3.0, 4.0])
    input = torch.sparse_coo_tensor(indices_input, values_input, (2, 2)).to_dense().numpy()

    input_dict = {
        "input": input,
        "mat1": mat1,
        "mat2": mat2,
        "beta": 1.0,
        "alpha": 1.0,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.sspaddmm"] = sspaddmm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sspaddmm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sspaddmm'.")

check_valid('torch.sspaddmm', generated_inputs['torch.sspaddmm'], lib="torch")
