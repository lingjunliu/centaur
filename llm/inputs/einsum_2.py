
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def einsum_inputs():
    list_of_inputs = []

    def to_numpy(arg):
        if isinstance(arg, np.ndarray):
            return arg
        return np.array(arg).astype(np.float32)

    # Input 1: Trace of a matrix
    A = np.random.rand(3, 3).astype(np.float32)
    operands = ['ii', A]
    input_dict = {"operands": [A]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Matrix multiplication
    A = np.random.rand(2, 3).astype(np.float32)
    B = np.random.rand(3, 4).astype(np.float32)
    operands = ['ij,jk->ik', A, B]
    input_dict = {"operands": [A, B]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Dot product of two vectors
    a = np.random.rand(5).astype(np.float32)
    b = np.random.rand(5).astype(np.float32)
    operands = ['i,i->', a, b]
    input_dict = {"operands": [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Batch matrix multiplication
    A = np.random.rand(2, 3, 4).astype(np.float32)
    B = np.random.rand(2, 4, 5).astype(np.float32)
    operands = ['bij,bjk->bik', A, B]
    input_dict = {"operands": [A, B]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Scalar multiplication
    A = np.random.rand(1).astype(np.float32)
    B = np.random.rand(1).astype(np.float32)
    operands = ['', A, B] # empty equation is for scalars
    input_dict = {"operands": [A, B]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.einsum_2"] = einsum_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.einsum_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.einsum_2'.")

check_valid('torch.einsum', generated_inputs['torch.einsum_2'], lib="torch", suffix=2)
