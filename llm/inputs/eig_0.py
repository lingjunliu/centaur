
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_linalg_eig_inputs():
    list_of_inputs = []

    # Input 1: Simple 2x2 real matrix
    A = torch.randn(2, 2).numpy()
    input_dict = {"A": A, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Batch of 3x3 real matrices
    A = torch.randn(3, 3, 3).numpy()
    input_dict = {"A": A, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Complex 2x2 matrix
    A = torch.randn(2, 2, dtype=torch.complex128).numpy()
    input_dict = {"A": A, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Batch of 2x2 complex matrices
    A = torch.randn(4, 2, 2, dtype=torch.complex128).numpy()
    input_dict = {"A": A, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4x4 real matrix
    A = torch.randn(4, 4).numpy()
    input_dict = {"A": A, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.linalg.eig"] = torch_linalg_eig_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.eig' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.eig'.")

check_valid('torch.linalg.eig', generated_inputs['torch.linalg.eig'], lib="torch", suffix=0)
