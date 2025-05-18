
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def cholesky_inverse_inputs():
    list_of_inputs = []

    # Example 1: Basic Cholesky inverse
    A = torch.randn(3, 3, dtype=torch.float64)
    A = A @ A.T
    A += torch.eye(3, dtype=torch.float64) * 1e-3  # Ensure positive definite
    L = torch.linalg.cholesky(A).numpy()
    input_dict = {"L": L, "upper": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Upper triangular Cholesky inverse
    A = torch.randn(4, 4, dtype=torch.float64)
    A = A @ A.T
    A += torch.eye(4, dtype=torch.float64) * 1e-3  # Ensure positive definite
    U = torch.linalg.cholesky(A).numpy()
    U = np.triu(U)
    input_dict = {"L": U.T, "upper": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: Batched Cholesky inverse
    A = torch.randn(2, 2, 2, dtype=torch.float64)
    A = A @ A.transpose(-1, -2)
    A += torch.eye(2, dtype=torch.float64).unsqueeze(0) * 1e-3  # Ensure positive definite
    L = torch.linalg.cholesky(A).numpy()
    input_dict = {"L": L, "upper": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: Larger matrix
    A = torch.randn(5, 5, dtype=torch.float64)
    A = A @ A.T
    A += torch.eye(5, dtype=torch.float64) * 1e-3  # Ensure positive definite
    L = torch.linalg.cholesky(A).numpy()
    input_dict = {"L": L, "upper": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5: Upper triangular, batched
    A = torch.randn(3, 3, 3, dtype=torch.float64)
    A = A @ A.transpose(-1, -2)
    A += torch.eye(3, dtype=torch.float64).unsqueeze(0) * 1e-3  # Ensure positive definite
    U = torch.linalg.cholesky(A).numpy()
    U = np.triu(U)
    input_dict = {"L": U.transpose(0, 2, 1), "upper": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = cholesky_inverse_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('cholesky_inverse', list_of_inputs)
