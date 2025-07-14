
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def linalg_cholesky_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    A = torch.tensor([[4.0, 1.0], [1.0, 4.25]], dtype=torch.float32)
    upper = False
    out = torch.zeros_like(A, dtype=torch.float32).numpy()
    input_dict = {"A": A.numpy(), "upper": upper, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: upper=True
    A = torch.tensor([[4.0, 1.0], [1.0, 4.25]], dtype=torch.float32)
    upper = True
    out = torch.zeros_like(A, dtype=torch.float32).numpy()
    input_dict = {"A": A.numpy(), "upper": upper, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: different shape
    A = torch.tensor([[16.0, 4.0], [4.0, 5.0]], dtype=torch.float32)
    upper = False
    out = torch.zeros_like(A, dtype=torch.float32).numpy()
    input_dict = {"A": A.numpy(), "upper": upper, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: larger matrix
    A = torch.tensor([[25.0, 5.0, 0.0], [5.0, 26.0, 1.0], [0.0, 1.0, 10.0]], dtype=torch.float32)
    upper = False
    out = torch.zeros_like(A, dtype=torch.float32).numpy()
    input_dict = {"A": A.numpy(), "upper": upper, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: batch of matrices
    A = torch.tensor([[[4.0, 1.0], [1.0, 4.25]], [[16.0, 4.0], [4.0, 5.0]]], dtype=torch.float32)
    upper = False
    out = torch.zeros_like(A, dtype=torch.float32).numpy()
    input_dict = {"A": A.numpy(), "upper": upper, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: double dtype
    A = torch.tensor([[4.0, 1.0], [1.0, 4.25]], dtype=torch.float64)
    upper = False
    out = torch.zeros_like(A, dtype=torch.float64).numpy()
    input_dict = {"A": A.numpy(), "upper": upper, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: complex dtype
    A = torch.tensor([[4.0 + 0j, 1.0 + 0j], [1.0 + 0j, 4.25 + 0j]], dtype=torch.complex128)
    upper = False
    out = torch.zeros_like(A, dtype=torch.complex128).numpy()
    input_dict = {"A": A.numpy(), "upper": upper, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float dtype
    A = torch.tensor([[4.0, 1.0], [1.0, 4.25]], dtype=torch.float32)
    upper = False
    out = torch.zeros_like(A, dtype=torch.float32).numpy()
    input_dict = {"A": A.numpy(), "upper": upper, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D tensor, upper=True
    A = torch.randn(2, 2, 2, dtype=torch.float64)
    A = A @ torch.transpose(A, 1, 2) + torch.eye(2)
    upper = True
    out = torch.zeros_like(A, dtype=torch.float64).numpy()
    input_dict = {"A": A.numpy(), "upper": upper, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: cdouble dtype
    A = torch.tensor([[4.0 + 0j, 1.0 + 0j], [1.0 + 0j, 4.25 + 0j]], dtype=torch.complex128)
    upper = True
    out = torch.zeros_like(A, dtype=torch.complex128).numpy()
    input_dict = {"A": A.numpy(), "upper": upper, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.linalg.cholesky"] = linalg_cholesky_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.cholesky' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.cholesky'.")

check_valid('torch.linalg.cholesky', generated_inputs['torch.linalg.cholesky'], lib="torch", suffix=0)
