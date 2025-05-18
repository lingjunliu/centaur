
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def narrow_inputs():
    list_of_inputs = []

    input = torch.randn(5, 5).numpy()
    dim = 0
    start = 1
    length = 3
    input_dict = {
        "input": input,
        "dim": dim,
        "start": start,
        "length": length
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(3, 4, 5).numpy()
    dim = 1
    start = 0
    length = 2
    input_dict = {
        "input": input,
        "dim": dim,
        "start": start,
        "length": length
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 6).numpy()
    dim = 1
    start = 2
    length = 3
    input_dict = {
        "input": input,
        "dim": dim,
        "start": start,
        "length": length
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(10).numpy()
    dim = 0
    start = 5
    length = 4
    input_dict = {
        "input": input,
        "dim": dim,
        "start": start,
        "length": length
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 3, 4, 5).numpy()
    dim = 2
    start = 1
    length = 2
    input_dict = {
        "input": input,
        "dim": dim,
        "start": start,
        "length": length
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = narrow_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('narrow', list_of_inputs)
