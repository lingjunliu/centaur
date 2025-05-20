
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def smm_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float32
    indices = torch.tensor([[0, 0], [1, 2], [2, 1]], dtype=torch.long)
    values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    size = torch.Size([3, 3])
    input_sparse = torch.sparse_coo_tensor(indices, values, size).to_dense()
    mat = torch.randn(3, 4).numpy()
    input_dict = {"input": input_sparse.numpy(), "mat": mat}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer sparse matrix and float dense matrix
    indices = torch.tensor([[0, 1], [1, 0], [2, 2]], dtype=torch.long)
    values = torch.tensor([4, 5, 6], dtype=torch.int64)
    size = torch.Size([3, 3])
    input_sparse = torch.sparse_coo_tensor(indices, values, size).to_dense()
    mat = torch.randn(3, 5).numpy()
    input_dict = {"input": input_sparse.numpy(), "mat": mat}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative values in sparse matrix
    indices = torch.tensor([[0, 0], [1, 1], [2, 2]], dtype=torch.long)
    values = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.float32)
    size = torch.Size([3, 3])
    input_sparse = torch.sparse_coo_tensor(indices, values, size).to_dense()
    mat = torch.randn(3, 2).numpy()
    input_dict = {"input": input_sparse.numpy(), "mat": mat}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Larger sparse matrix and smaller dense matrix (transpose needed)
    indices = torch.tensor([[0, 0], [1, 2], [2, 1], [3, 3]], dtype=torch.long)
    values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    size = torch.Size([4, 4])
    input_sparse = torch.sparse_coo_tensor(indices, values, size).to_dense()
    mat = torch.randn(4, 3).numpy()
    input_dict = {"input": input_sparse.numpy(), "mat": mat}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Rectangular sparse matrix
    indices = torch.tensor([[0, 0], [1, 1], [2, 0]], dtype=torch.long)
    values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    size = torch.Size([3, 2])
    input_sparse = torch.sparse_coo_tensor(indices, values, size).to_dense()
    mat = torch.randn(2, 4).numpy()
    input_dict = {"input": input_sparse.numpy(), "mat": mat}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = smm_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('smm', generated_inputs)
