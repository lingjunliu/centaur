
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def einsum_inputs():
    list_of_inputs = []

    # Input 1: Matrix multiplication
    A = np.random.rand(2, 3).astype(np.float32)
    B = np.random.rand(3, 4).astype(np.float32)
    operands = ['ij,jk->ik', A, B]
    list_of_inputs.append({'operands': operands})

    # Input 2: Trace of a matrix
    A = np.random.rand(5, 5).astype(np.float32)
    operands = ['ii', A]
    list_of_inputs.append({'operands': operands})

    # Input 3: Batch matrix multiplication
    A = np.random.rand(2, 3, 4).astype(np.float32)
    B = np.random.rand(2, 4, 5).astype(np.float32)
    operands = ['bij,bjk->bik', A, B]
    list_of_inputs.append({'operands': operands})

    # Input 4: Dot product of two vectors
    A = np.random.rand(6).astype(np.float32)
    B = np.random.rand(6).astype(np.float32)
    operands = ['i,i', A, B]
    list_of_inputs.append({'operands': operands})

    # Input 5: Transpose of a matrix
    A = np.random.rand(4, 2).astype(np.float32)
    operands = ['ij->ji', A]
    list_of_inputs.append({'operands': operands})

    # Input 6: Outer product of two vectors
    A = np.random.rand(3).astype(np.float32)
    B = np.random.rand(5).astype(np.float32)
    operands = ['i,j->ij', A, B]
    list_of_inputs.append({'operands': operands})

    # Input 7: Sum along an axis - Reducing dimensions
    A = np.random.rand(2, 3, 4).astype(np.float32)
    operands = ['ijk->k', A]
    list_of_inputs.append({'operands': operands})

    # Input 8: Bilinear transformation
    A = np.random.rand(5, 3).astype(np.float32) # Modified shape
    B = np.random.rand(2, 5).astype(np.float32)
    C = np.random.rand(2, 3).astype(np.float32)
    operands = ['bn,an,bm->am', B, A, C] # Modified equation
    list_of_inputs.append({'operands': operands})

    # Input 9: Diagonal elements
    A = np.random.rand(4, 4).astype(np.float32)
    operands = ['ii->i', A]
    list_of_inputs.append({'operands': operands})

    # Input 10: With ellipsis - Simplified
    A = np.random.rand(2, 3).astype(np.float32)
    operands = ['...ij->...ji', A]
    list_of_inputs.append({'operands': operands})
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.einsum_2"] = einsum_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.einsum_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.einsum_2'.")

check_valid('torch.einsum', generated_inputs['torch.einsum_2'], lib="torch", suffix=2)
