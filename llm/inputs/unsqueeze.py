
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def unsqueeze_inputs():
    list_of_inputs = []

    input_tensor = torch.randn(2, 3).numpy()
    dim = 0
    input_dict = {
        "tensor": input_tensor,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(2, 3, 4).numpy()
    dim = 1
    input_dict = {
        "tensor": input_tensor,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(5).numpy()
    dim = 0
    input_dict = {
        "tensor": input_tensor,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(2, 2, 2, 2).numpy()
    dim = 3
    input_dict = {
        "tensor": input_tensor,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(3, 4, 5).numpy()
    dim = -1
    input_dict = {
        "tensor": input_tensor,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = unsqueeze_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('unsqueeze', list_of_inputs)
