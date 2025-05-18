
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def linear_inputs():
    list_of_inputs = []

    input = torch.randn(3, 5).numpy()
    weight = torch.randn(2, 5).numpy()
    bias = torch.randn(2).numpy()

    input_dict = {
        "input": input,
        "weight": weight,
        "bias": bias
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 10).numpy()
    weight = torch.randn(5, 10).numpy()
    bias = torch.randn(5).numpy()

    input_dict = {
        "input": input,
        "weight": weight,
        "bias": bias
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 4).numpy()
    weight = torch.randn(3, 4).numpy()
    bias = torch.randn(3).numpy()

    input_dict = {
        "input": input,
        "weight": weight,
        "bias": bias
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(5, 7).numpy()
    weight = torch.randn(1, 7).numpy()
    bias = torch.randn(1).numpy()

    input_dict = {
        "input": input,
        "weight": weight,
        "bias": bias
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(4, 6).numpy()
    weight = torch.randn(8, 6).numpy()
    bias = torch.randn(8).numpy()

    input_dict = {
        "input": input,
        "weight": weight,
        "bias": bias
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = linear_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('linear_', list_of_inputs)
