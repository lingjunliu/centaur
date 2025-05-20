
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def torch_linalg_pinv_inputs():
    generated_inputs = []

    # Input 1: Basic float tensor
    A = torch.randn(3, 5).numpy()
    input_dict = {"A": A, "rtol": 1e-5, "atol": 1e-8, "hermitian": False}
    generated_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Batch of matrices, different rtol and atol
    A = torch.randn(2, 6, 3).numpy()
    input_dict = {"A": A, "rtol": 1e-3, "atol": 1e-6, "hermitian": False}
    generated_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Complex Hermitian matrix
    A = torch.randn(3, 3, dtype=torch.complex64)
    A = (A + A.T.conj()).numpy()
    input_dict = {"A": A, "rtol": 1e-5, "atol": 1e-8, "hermitian": True}
    generated_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Double precision tensor with specified atol only
    A = torch.randn(4, 4, dtype=torch.float64).numpy()
    input_dict = {"A": A, "rtol": None, "atol": 1e-9, "hermitian": False}
    generated_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Single value matrix with negative value
    A = torch.tensor([[-2.0]]).numpy()
    input_dict = {"A": A, "rtol": 1e-5, "atol": 1e-8, "hermitian": False}
    generated_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different shape matrix
    A = torch.randn(5, 2).numpy()
    input_dict = {"A": A, "rtol": 1e-5, "atol": 1e-8, "hermitian": False}
    generated_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Matrix of zeros
    A = torch.zeros(2, 3).numpy()
    input_dict = {"A": A, "rtol": 1e-5, "atol": 1e-8, "hermitian": False}
    generated_inputs.append(copy.deepcopy(input_dict))

    return generated_inputs

generated_inputs = torch_linalg_pinv_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('pinv', generated_inputs)
