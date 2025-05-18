
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy

def prelu_inputs():
    list_of_inputs = []

    input = torch.randn(3, 4, 5).numpy()
    weight = torch.randn(1).numpy()
    input_dict = {
        "input": input,
        "weight": weight
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 3, 4, 5).numpy()
    weight = torch.randn(1).numpy()
    input_dict = {
        "input": input,
        "weight": weight
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 5).numpy()
    weight = torch.randn(1).numpy()
    input_dict = {
        "input": input,
        "weight": weight
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 2).numpy()
    weight = torch.randn(1).numpy()
    input_dict = {
        "input": input,
        "weight": weight
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(4).numpy()
    weight = torch.randn(1).numpy()
    input_dict = {
        "input": input,
        "weight": weight
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = prelu_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('prelu', list_of_inputs)
