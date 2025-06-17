
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_einsum_inputs():
    list_of_inputs = []

    # Example 1: Matrix multiplication
    A = np.random.randn(3, 4).astype(np.float32)
    B = np.random.randn(4, 5).astype(np.float32)
    input_dict = {
        "operands": ['ij,jk->ik', torch.from_numpy(A), torch.from_numpy(B)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Batch matrix multiplication
    A = np.random.randn(2, 3, 4).astype(np.float32)
    B = np.random.randn(2, 4, 5).astype(np.float32)
    input_dict = {
        "operands": ['bij,bjk->bik', torch.from_numpy(A), torch.from_numpy(B)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: Trace of a matrix
    A = np.random.randn(5, 5).astype(np.float32)
    input_dict = {
        "operands": ['ii', torch.from_numpy(A)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.einsum_2"] = torch_einsum_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.einsum_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.einsum_2'.")

check_valid('torch.einsum', generated_inputs['torch.einsum_2'], lib="torch")
