
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def dot_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensors
    a = torch.randn(3).numpy()
    b = torch.randn(3).numpy()
    input_dict = {"input": a, "tensor": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Integer tensors
    a = torch.randint(0, 10, (4,)).numpy()
    b = torch.randint(0, 10, (4,)).numpy()
    input_dict = {"input": a, "tensor": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Negative values
    a = torch.randint(-10, 0, (5,)).numpy()
    b = torch.randint(-5, 5, (5,)).numpy()
    input_dict = {"input": a, "tensor": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Large tensors
    a = torch.randn(1000).numpy()
    b = torch.randn(1000).numpy()
    input_dict = {"input": a, "tensor": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Small tensors
    a = torch.randn(1).numpy()
    b = torch.randn(1).numpy()
    input_dict = {"input": a, "tensor": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Float64
    a = torch.randn(3, dtype=torch.float64).numpy()
    b = torch.randn(3, dtype=torch.float64).numpy()
    input_dict = {"input": a, "tensor": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = dot_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('dot', generated_inputs)
