
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def split_inputs():
    list_of_inputs = []

    tensor = torch.randn(4, 4).numpy()
    split_size_or_sections = 2
    dim = 0
    input_dict = {
        "tensor": tensor,
        "split_size_or_sections": split_size_or_sections,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = torch.randn(6, 6).numpy()
    split_size_or_sections = 3
    dim = 1
    input_dict = {
        "tensor": tensor,
        "split_size_or_sections": split_size_or_sections,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = torch.randn(2, 8, 4).numpy()
    split_size_or_sections = 4
    dim = 1
    input_dict = {
        "tensor": tensor,
        "split_size_or_sections": split_size_or_sections,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = torch.randn(5, 5, 5).numpy()
    split_size_or_sections = 1
    dim = 2
    input_dict = {
        "tensor": tensor,
        "split_size_or_sections": split_size_or_sections,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = torch.randn(10).numpy()
    split_size_or_sections = 5
    dim = 0
    input_dict = {
        "tensor": tensor,
        "split_size_or_sections": split_size_or_sections,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

list_of_inputs = split_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('split', list_of_inputs)
