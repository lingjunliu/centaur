
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def crow_indices_copy_inputs():
    generated_inputs = []

    dense_tensor1 = torch.tensor([[0, 1, 0], [2, 0, 3]])
    sparse_tensor1 = dense_tensor1.to_sparse_csr()
    input1 = sparse_tensor1.crow_indices().numpy()
    generated_inputs.append({"input": input1})

    dense_tensor2 = torch.tensor([[0, 1, 0], [2, 0, 3]], dtype=torch.int64)
    sparse_tensor2 = dense_tensor2.to_sparse_csr()
    input2 = sparse_tensor2.crow_indices().numpy()
    generated_inputs.append({"input": input2})

    dense_tensor3 = torch.tensor([[0, 1, 0], [2, 0, 3]], dtype=torch.float32)
    sparse_tensor3 = dense_tensor3.to_sparse_csr()
    input3 = sparse_tensor3.crow_indices().numpy()
    generated_inputs.append({"input": input3})
    
    dense_tensor6 = torch.tensor([[1, 2], [3, 4]])
    sparse_tensor6 = dense_tensor6.to_sparse_csr()
    input6 = sparse_tensor6.crow_indices().numpy()
    generated_inputs.append({"input": input6})

    return generated_inputs

generated_inputs = crow_indices_copy_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('crow_indices_copy', generated_inputs)
