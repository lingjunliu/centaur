
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def smm_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float sparse and dense matrices
    indices = torch.tensor([[0, 0], [1, 2], [2, 1]], dtype=torch.long)
    values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float)
    size = torch.Size([3, 3])
    sparse_input = torch.sparse_coo_tensor(indices, values, size).coalesce()
    dense_mat = torch.randn(3, 4).numpy()
    input_dict = {"input": sparse_input.to_dense().numpy(), "mat": dense_mat}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different dimensions, int sparse and dense matrices
    indices = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=torch.long)
    values = torch.tensor([1, 2, 3, 4], dtype=torch.int)
    size = torch.Size([4, 4])
    sparse_input = torch.sparse_coo_tensor(indices, values, size).coalesce()
    dense_mat = torch.randint(0, 5, (4, 2)).numpy()
    input_dict = {"input": sparse_input.to_dense().numpy(), "mat": dense_mat}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative values, float sparse and dense matrices
    indices = torch.tensor([[0, 1], [1, 0], [2, 2]], dtype=torch.long)
    values = torch.tensor([-1.0, 2.0, -3.0], dtype=torch.float)
    size = torch.Size([3, 3])
    sparse_input = torch.sparse_coo_tensor(indices, values, size).coalesce()
    dense_mat = torch.randn(3, 5).numpy()
    input_dict = {"input": sparse_input.to_dense().numpy(), "mat": dense_mat}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Larger sparse matrix, complex dense matrix
    indices = torch.tensor([[0, 0], [1, 2], [3, 1], [4,4]], dtype=torch.long)
    values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float)
    size = torch.Size([5, 5])
    sparse_input = torch.sparse_coo_tensor(indices, values, size).coalesce()
    dense_mat = (torch.randn(5, 3) + 1j * torch.randn(5, 3)).numpy()
    input_dict = {"input": sparse_input.to_dense().numpy(), "mat": dense_mat}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Rectangular matrices, mixed types
    indices = torch.tensor([[0, 0], [1, 2], [2, 0]], dtype=torch.long)
    values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float)
    size = torch.Size([3, 4])
    sparse_input = torch.sparse_coo_tensor(indices, values, size).coalesce()
    dense_mat = torch.randn(4, 2).numpy()
    input_dict = {"input": sparse_input.to_dense().numpy(), "mat": dense_mat}
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
