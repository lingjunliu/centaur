
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def matrix_rank_inputs():
    list_of_inputs = []

    A = torch.randn(3, 3).numpy()
    tol = 1e-8
    atol = 1e-8
    rtol = 1e-5
    hermitian = False
    input_dict = {
        "A": A,
        "tol": float(tol),
        "atol": float(atol),
        "rtol": float(rtol),
        "hermitian": hermitian
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = torch.randn(5, 5).numpy()
    tol = 1e-5
    atol = 1e-5
    rtol = 1e-3
    hermitian = True
    input_dict = {
        "A": A,
        "tol": float(tol),
        "atol": float(atol),
        "rtol": float(rtol),
        "hermitian": hermitian
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    tol = 1e-6
    atol = 1e-6
    rtol = 1e-4
    hermitian = False
    input_dict = {
        "A": A,
        "tol": float(tol),
        "atol": float(atol),
        "rtol": float(rtol),
        "hermitian": hermitian
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = torch.randn(2, 4).numpy()
    tol = 1e-7
    atol = 1e-7
    rtol = 1e-5
    hermitian = False
    input_dict = {
        "A": A,
        "tol": float(tol),
        "atol": float(atol),
        "rtol": float(rtol),
        "hermitian": hermitian
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = np.eye(4).astype(np.float32)
    tol = 1e-8
    atol = 1e-8
    rtol = 1e-6
    hermitian = True
    input_dict = {
        "A": A,
        "tol": float(tol),
        "atol": float(atol),
        "rtol": float(rtol),
        "hermitian": hermitian
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = matrix_rank_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('matrix_rank', list_of_inputs)
