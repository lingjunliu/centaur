
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def addcdiv_inputs():
    list_of_inputs = []

    input = torch.randn(3, 4).numpy()
    tensor1 = torch.randn(3, 4).numpy()
    tensor2 = torch.randn(3, 4).numpy()
    value = 2.0

    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 2, 2).numpy()
    tensor1 = torch.randn(2, 2, 2).numpy()
    tensor2 = torch.randn(2, 2, 2).numpy()
    value = 0.5

    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(5).numpy()
    tensor1 = torch.randn(5).numpy()
    tensor2 = torch.randn(5).numpy()
    value = -1.0

    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 5, 5).numpy()
    tensor1 = torch.randn(1, 5, 5).numpy()
    tensor2 = torch.randn(1, 5, 5).numpy()
    value = 1.5

    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 3, 4, 5).numpy()
    tensor1 = torch.randn(2, 3, 4, 5).numpy()
    tensor2 = torch.randn(2, 3, 4, 5).numpy()
    value = 0.0

    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = addcdiv_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('addcdiv', list_of_inputs)
