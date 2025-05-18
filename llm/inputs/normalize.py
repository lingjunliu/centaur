
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def normalize_inputs():
    list_of_inputs = []

    input1 = torch.randn(3, 5).numpy()
    dim1 = 0
    eps1 = 1e-12
    p1 = 2.0

    input_dict1 = {
        "input": input1,
        "dim": dim1,
        "eps": eps1,
        "p": p1
    }

    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 4, 6).numpy()
    dim2 = 1
    eps2 = 1e-8
    p2 = 1.0

    input_dict2 = {
        "input": input2,
        "dim": dim2,
        "eps": eps2,
        "p": p2
    }

    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(10).numpy()
    dim3 = 0
    eps3 = 1e-5
    p3 = float('inf')

    input_dict3 = {
        "input": input3,
        "dim": dim3,
        "eps": eps3,
        "p": p3
    }

    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(2, 2, 2).numpy()
    dim4 = 2
    eps4 = 1e-6
    p4 = -2.0

    input_dict4 = {
        "input": input4,
        "dim": dim4,
        "eps": eps4,
        "p": p4
    }

    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(1, 5, 7, 9).numpy()
    dim5 = 3
    eps5 = 1e-4
    p5 = 0.5

    input_dict5 = {
        "input": input5,
        "dim": dim5,
        "eps": eps5,
        "p": p5
    }

    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

list_of_inputs = normalize_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('normalize', list_of_inputs)
