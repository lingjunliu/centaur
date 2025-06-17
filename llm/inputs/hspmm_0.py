
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def hspmm_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float tensors
    indices = torch.tensor([[0, 1], [1, 0]])
    values = torch.tensor([1.0, 2.0])
    size = (2, 2)
    mat1 = torch.sparse_coo_tensor(indices, values, size).coalesce()
    mat2 = torch.randn(2, 3).numpy()
    input_dict = {"mat1": mat1, "mat2": mat2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer tensors
    indices = torch.tensor([[0, 1], [1, 0]])
    values = torch.tensor([1, 2])
    size = (2, 2)
    mat1 = torch.sparse_coo_tensor(indices, values, size).coalesce()
    mat2 = torch.randint(0, 5, (2, 3)).numpy()
    input_dict = {"mat1": mat1, "mat2": mat2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different dimensions
    indices = torch.tensor([[0, 1], [1, 2]])
    values = torch.tensor([1.0, 2.0])
    size = (2, 3)
    mat1 = torch.sparse_coo_tensor(indices, values, size).coalesce()
    mat2 = torch.randn(3, 4).numpy()
    input_dict = {"mat1": mat1, "mat2": mat2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative values
    indices = torch.tensor([[0, 1], [1, 0]])
    values = torch.tensor([-1.0, 2.0])
    size = (2, 2)
    mat1 = torch.sparse_coo_tensor(indices, values, size).coalesce()
    mat2 = torch.randn(2, 3).numpy()
    input_dict = {"mat1": mat1, "mat2": mat2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger matrices
    indices = torch.tensor([[0, 1], [1, 2]])
    values = torch.tensor([1.0, 2.0])
    size = (3, 3)
    mat1 = torch.sparse_coo_tensor(indices, values, size).coalesce()
    mat2 = torch.randn(3, 5).numpy()
    input_dict = {"mat1": mat1, "mat2": mat2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: With out tensor. The out tensor must be sparse
    indices = torch.tensor([[0, 1], [1, 0]])
    values = torch.tensor([1.0, 2.0])
    size = (2, 2)
    mat1 = torch.sparse_coo_tensor(indices, values, size).coalesce()
    mat2 = torch.randn(2, 3).numpy()
    out = torch.sparse_coo_tensor(torch.empty(0, 2, dtype=torch.long), torch.empty(0), (2, 3)).coalesce()
    input_dict = {"mat1": mat1, "mat2": mat2, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.hspmm"] = hspmm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.hspmm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.hspmm'.")

check_valid('torch.hspmm', generated_inputs['torch.hspmm'], lib="torch")
