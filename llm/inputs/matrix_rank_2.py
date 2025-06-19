
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def matrix_rank_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D matrix
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    tol = torch.tensor([1e-8]).numpy()
    hermitian = False
    input_dict = {"input": input, "tol": tol, "hermitian": hermitian}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Singular matrix
    input = torch.tensor([[1.0, 2.0], [2.0, 4.0]]).numpy()
    tol = torch.tensor([1e-8]).numpy()
    hermitian = False
    input_dict = {"input": input, "tol": tol, "hermitian": hermitian}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different tolerance
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    tol = torch.tensor([0.1]).numpy()
    hermitian = False
    input_dict = {"input": input, "tol": tol, "hermitian": hermitian}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Complex input, hermitian=True
    input = torch.tensor([[1.0, 2.0 + 1j], [2.0 - 1j, 3.0]]).numpy()
    tol = torch.tensor([1e-8]).numpy()
    hermitian = True
    input_dict = {"input": input, "tol": tol, "hermitian": hermitian}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Rectangular matrix
    input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).numpy()
    tol = torch.tensor([1e-8]).numpy()
    hermitian = False
    input_dict = {"input": input, "tol": tol, "hermitian": hermitian}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.linalg.matrix_rank_2"] = matrix_rank_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.matrix_rank_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.matrix_rank_2'.")

check_valid('torch.linalg.matrix_rank', generated_inputs['torch.linalg.matrix_rank_2'], lib="torch", suffix=2)
