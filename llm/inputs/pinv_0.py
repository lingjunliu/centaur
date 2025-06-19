
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def linalg_pinv_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensor
    A = torch.randn(3, 5).numpy()
    input_dict = {"A": A, "atol": None, "rtol": None, "hermitian": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Batch of matrices
    A = torch.randn(2, 6, 3).numpy()
    input_dict = {"A": A, "atol": None, "rtol": None, "hermitian": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Hermitian matrix (complex)
    A = torch.randn(3, 3, dtype=torch.complex64)
    A = A + A.T.conj()
    A = A.numpy()
    input_dict = {"A": A, "atol": None, "rtol": None, "hermitian": True, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: with atol and rtol
    A = torch.randn(4, 4).numpy()
    atol = 1e-6
    rtol = 1e-5
    input_dict = {"A": A, "atol": atol, "rtol": rtol, "hermitian": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Non-square matrix with atol
    A = torch.randn(5, 3).numpy()
    atol = 1e-4
    input_dict = {"A": A, "atol": atol, "rtol": None, "hermitian": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.linalg.pinv"] = linalg_pinv_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.pinv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.pinv'.")

check_valid('torch.linalg.pinv', generated_inputs['torch.linalg.pinv'], lib="torch")
