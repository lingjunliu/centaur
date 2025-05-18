
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def where_inputs():
    list_of_inputs = []

    condition = (torch.randn(3, 2) > 0).numpy()
    input_tensor = torch.randn(3, 2).numpy()
    other_tensor = torch.randn(3, 2).numpy()
    input_dict = {
        "condition": condition,
        "input": input_tensor,
        "other": other_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    condition = (torch.randn(2, 2, 2) > 0).numpy()
    input_tensor = torch.randn(2, 2, 2).numpy()
    other_tensor = torch.randn(2, 2, 2).numpy()
    input_dict = {
        "condition": condition,
        "input": input_tensor,
        "other": other_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    condition = (torch.randn(5) > 0).numpy()
    input_tensor = torch.randn(5).numpy()
    other_tensor = torch.randn(5).numpy()
    input_dict = {
        "condition": condition,
        "input": input_tensor,
        "other": other_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    condition = (torch.randn(1, 4, 4) > 0).numpy()
    input_tensor = torch.randn(1, 4, 4).numpy()
    other_tensor = torch.randn(1, 4, 4).numpy()
    input_dict = {
        "condition": condition,
        "input": input_tensor,
        "other": other_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    condition = (torch.randn(2, 3, 4, 5) > 0).numpy()
    input_tensor = torch.randn(2, 3, 4, 5).numpy()
    other_tensor = torch.randn(2, 3, 4, 5).numpy()
    input_dict = {
        "condition": condition,
        "input": input_tensor,
        "other": other_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = where_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('where', list_of_inputs)
