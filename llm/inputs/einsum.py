
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def einsum_inputs():
    list_of_inputs = []

    # Case 1: Matrix multiplication
    a = np.random.rand(2, 3).astype(np.float32)
    b = np.random.rand(3, 4).astype(np.float32)
    input_dict = {
        'equation': 'ij,jk->ik',
        'operands': [a, b]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Batch matrix multiplication
    a = np.random.rand(5, 2, 3).astype(np.float32)
    b = np.random.rand(5, 3, 4).astype(np.float32)
    input_dict = {
        'equation': 'bij,bjk->bik',
        'operands': [a, b]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Trace of a matrix
    a = np.random.rand(4, 4).astype(np.float32)
    input_dict = {
        'equation': 'ii->',
        'operands': [a]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Sum along an axis
    a = np.random.rand(3, 5).astype(np.float32)
    input_dict = {
        'equation': 'ij->i',
        'operands': [a]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 5: Transpose
    a = np.random.rand(3, 5).astype(np.float32)
    input_dict = {
        'equation': 'ij->ji',
        'operands': [a]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Dot product
    a = np.random.rand(5).astype(np.float32)
    b = np.random.rand(5).astype(np.float32)
    input_dict = {
        'equation': 'i,i->',
        'operands': [a, b]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Hadamard product and sum
    a = np.random.rand(3, 3).astype(np.float32)
    b = np.random.rand(3, 3).astype(np.float32)
    input_dict = {
        'equation': 'ij,ij->',
        'operands': [a, b]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: More complex with 3 tensors
    a = np.random.rand(2, 3).astype(np.float32)
    b = np.random.rand(3, 4).astype(np.float32)
    c = np.random.rand(4, 2).astype(np.float32)
    input_dict = {
        'equation': 'ij,jk,kl->il',
        'operands': [a, b, c]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 9: Broadcasting Example
    a = np.random.rand(3, 1).astype(np.float32)
    b = np.random.rand(1, 4).astype(np.float32)
    input_dict = {
        'equation': 'ij,jk->ik',
        'operands': [a, b]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 10: Integer tensors
    a = np.random.randint(0, 10, size=(2, 3)).astype(np.int32)
    b = np.random.randint(0, 10, size=(3, 4)).astype(np.int32)
    input_dict = {
        'equation': 'ij,jk->ik',
        'operands': [a, b]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = einsum_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('einsum', generated_inputs)
