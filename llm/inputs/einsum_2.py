
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy, numpy as np

def torch_einsum_inputs():
    list_of_inputs = []

    # Example 1: Matrix multiplication
    A = torch.randn(3, 4).numpy()
    B = torch.randn(4, 5).numpy()
    input_dict = {
        "operands": ['ij,jk->ik', A, B]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Batch matrix multiplication
    As = torch.randn(2, 3, 4).numpy()
    Bs = torch.randn(2, 4, 5).numpy()
    input_dict = {
        "operands": ['bij,bjk->bik', As, Bs]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: Trace
    A = torch.randn(4, 4).numpy()
    input_dict = {
        "operands": ['ii', A]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: Outer product
    x = torch.randn(5).numpy()
    y = torch.randn(4).numpy()
    input_dict = {
        "operands": ['i,j->ij', x, y]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5: Diagonal
    A = torch.randn(4, 4).numpy()
    input_dict = {
        "operands": ['ii->i', A]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.einsum_2"] = torch_einsum_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.einsum_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.einsum_2'.")

check_valid('torch.einsum', generated_inputs['torch.einsum_2'], lib="torch")
