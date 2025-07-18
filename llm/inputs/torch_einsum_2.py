
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy as np

def einsum_inputs():
    list_of_inputs = []

    # Input 1: Trace of a matrix
    op1 = {
        'operands': [
            'ii',
            np.random.rand(5, 5).astype(np.float32)
        ]
    }
    list_of_inputs.append(copy.deepcopy(op1))

    # Input 2: Diagonal of a matrix
    op2 = {
        'operands': [
            'ii->i',
            np.random.rand(6, 6).astype(np.float32)
        ]
    }
    list_of_inputs.append(copy.deepcopy(op2))

    # Input 3: Sum over a dimension
    op3 = {
        'operands': [
            'ij->i',
            np.random.rand(4, 5).astype(np.float32)
        ]
    }
    list_of_inputs.append(copy.deepcopy(op3))

    # Input 4: Transpose a matrix
    op4 = {
        'operands': [
            'ij->ji',
            np.random.rand(7, 8).astype(np.float32)
        ]
    }
    list_of_inputs.append(copy.deepcopy(op4))

    # Input 5: Permute a 3D tensor
    op5 = {
        'operands': [
            'ijk->ikj',
            np.random.rand(2, 3, 4).astype(np.float32)
        ]
    }
    list_of_inputs.append(copy.deepcopy(op5))

    # Input 6: Batch permute with ellipsis
    op6 = {
        'operands': [
            '...ij->...ji',
            np.random.rand(10, 3, 4, 5).astype(np.float32)
        ]
    }
    list_of_inputs.append(copy.deepcopy(op6))

    # Input 7: Sum over all dimensions
    op7 = {
        'operands': [
            'ij->',
            np.random.rand(5, 5).astype(np.float64)
        ]
    }
    list_of_inputs.append(copy.deepcopy(op7))

    # Input 8: Matrix multiplication
    op8 = {
        'operands': [
            'ij,jk->ik',
            np.random.rand(3, 4).astype(np.float32),
            np.random.rand(4, 5).astype(np.float32)
        ]
    }
    list_of_inputs.append(copy.deepcopy(op8))
    
    # Input 9: Dot product
    op9 = {
        'operands': [
            'i,i->',
            np.random.rand(10).astype(np.float32),
            np.random.rand(10).astype(np.float32)
        ]
    }
    list_of_inputs.append(copy.deepcopy(op9))

    # Input 10: Outer product
    op10 = {
        'operands': [
            'i,j->ij',
            np.random.rand(7).astype(np.float32),
            np.random.rand(8).astype(np.float32)
        ]
    }
    list_of_inputs.append(copy.deepcopy(op10))


    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.einsum_2"] = einsum_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.einsum_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.einsum_2'.")

check_valid('torch.einsum', generated_inputs['torch.einsum_2'], lib="torch", suffix=2)
