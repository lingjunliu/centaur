
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def torch_linalg_tensorinv_inputs():
    generated_inputs = []

    # Case 1: Simple 4D tensor, ind=2
    A = torch.eye(4 * 6).reshape((4, 6, 8, 3)).numpy()
    input_dict = {"a": A, "ind": 2}
    generated_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 2D tensor, ind=1 (equivalent to torch.linalg.inv)
    A = torch.randn(4, 4).numpy()
    input_dict = {"a": A, "ind": 1}
    generated_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 3D tensor, ind=1
    A = torch.randn(6, 4, 4).numpy()
    A = A.reshape(4,6,4)
    input_dict = {"a": A, "ind": 1}
    generated_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Float64 tensor
    A = torch.eye(4).double().numpy()
    input_dict = {"a": A, "ind": 1}
    generated_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Complex tensor
    A = (torch.randn(2, 2) + 1j * torch.randn(2, 2)).numpy()
    input_dict = {"a": A, "ind": 1}
    generated_inputs.append(copy.deepcopy(input_dict))

    return generated_inputs

generated_inputs = torch_linalg_tensorinv_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('tensorinv', generated_inputs)
