
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def spmm_inputs():
    list_of_inputs = []

    # Input 1: Basic float sparse matrix and dense matrix
    sparse_mat = torch.randn(3, 4).to_sparse_coo().to_dense().numpy()
    dense_mat = torch.randn(4, 5).numpy()
    input_dict = {"input": sparse_mat, "mat2": dense_mat}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different sized matrices
    sparse_mat = torch.randn(5, 2).to_sparse_coo().to_dense().numpy()
    dense_mat = torch.randn(2, 3).numpy()
    input_dict = {"input": sparse_mat, "mat2": dense_mat}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Int sparse matrix and dense matrix
    sparse_mat = torch.randint(0, 10, (4, 3)).to_sparse_coo().to_dense().numpy()
    dense_mat = torch.randint(0, 10, (3, 2)).numpy()
    input_dict = {"input": sparse_mat, "mat2": dense_mat}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative values
    sparse_mat = torch.randn(2, 5).to_sparse_coo().to_dense().numpy()
    dense_mat = torch.randn(5, 4).numpy()
    input_dict = {"input": sparse_mat, "mat2": dense_mat}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger matrices
    sparse_mat = torch.randn(10, 20).to_sparse_coo().to_dense().numpy()
    dense_mat = torch.randn(20, 15).numpy()
    input_dict = {"input": sparse_mat, "mat2": dense_mat}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.spmm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.spmm'.")

check_valid('torch.spmm', generated_inputs['torch.spmm'], lib="torch")
