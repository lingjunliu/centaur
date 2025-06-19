
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_linalg_eigvals_inputs():
    list_of_inputs = []

    # Input 1: Basic real matrix
    A = torch.randn(2, 2).numpy()
    out = torch.zeros(2, dtype=torch.complex128).numpy()

    input_dict = {
        "A": A,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Complex matrix
    A = torch.randn(3, 3, dtype=torch.complex128).numpy()
    out = torch.zeros(3, dtype=torch.complex128).numpy()
    input_dict = {
        "A": A,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Batch of matrices
    A = torch.randn(2, 4, 4).numpy()
    out = torch.zeros(2, 4, dtype=torch.complex128).numpy()
    input_dict = {
        "A": A,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4:  Real matrix with pre-allocated out
    A = torch.randn(5, 5).numpy()
    out = torch.zeros(5, dtype=torch.complex128).numpy()

    input_dict = {
        "A": A,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Batch of complex matrices
    A = torch.randn(3, 2, 2, dtype=torch.complex64).numpy()
    out = torch.zeros(3, 2, dtype=torch.complex64).numpy()
    input_dict = {
        "A": A,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.linalg.eigvals"] = torch_linalg_eigvals_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.eigvals' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.eigvals'.")

check_valid('torch.linalg.eigvals', generated_inputs['torch.linalg.eigvals'], lib="torch", suffix=0)
