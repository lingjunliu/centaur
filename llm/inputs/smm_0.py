
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def smm_inputs():
    list_of_inputs = []

    # Case 1: Basic case with float tensors
    indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)
    values = torch.tensor([1.0, 2.0])
    size = (3, 3)
    input_sparse = torch.sparse_coo_tensor(indices, values, size).coalesce()
    mat = torch.randn(3, 2).numpy()
    input_dict = {"input": input_sparse.numpy(), "mat": mat}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Integer tensors
    indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)
    values = torch.tensor([1, 2])
    size = (3, 3)
    input_sparse = torch.sparse_coo_tensor(indices, values, size).coalesce()
    mat = torch.randint(0, 10, (3, 2)).numpy()
    input_dict = {"input": input_sparse.numpy(), "mat": mat}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Different dimensions for mat
    indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)
    values = torch.tensor([1.0, 2.0])
    size = (3, 3)
    input_sparse = torch.sparse_coo_tensor(indices, values, size).coalesce()
    mat = torch.randn(3, 5).numpy()
    input_dict = {"input": input_sparse.numpy(), "mat": mat}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 4: Negative values
    indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)
    values = torch.tensor([-1.0, 2.0])
    size = (3, 3)
    input_sparse = torch.sparse_coo_tensor(indices, values, size).coalesce()
    mat = torch.randn(3, 2).numpy()
    input_dict = {"input": input_sparse.numpy(), "mat": mat}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Larger sparse matrix
    indices = torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.int64)
    values = torch.tensor([1.0, 2.0, 3.0])
    size = (3, 3)
    input_sparse = torch.sparse_coo_tensor(indices, values, size).coalesce()
    mat = torch.randn(3, 4).numpy()
    input_dict = {"input": input_sparse.numpy(), "mat": mat}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.smm"] = smm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.smm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.smm'.")

check_valid('torch.smm', generated_inputs['torch.smm'], lib="torch")
