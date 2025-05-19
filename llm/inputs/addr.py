
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def addr_inputs():
    list_of_inputs = []

    input1 = torch.randn(3, 3).numpy()
    vec11 = torch.randn(3).numpy()
    vec21 = torch.randn(3).numpy()
    beta1 = 1.0
    alpha1 = 1.0
    input_dict1 = {
        "input": input1,
        "vec1": vec11,
        "vec2": vec21,
        "beta": beta1,
        "alpha": alpha1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.zeros(2, 4).numpy()
    vec12 = torch.ones(2).numpy()
    vec22 = torch.ones(4).numpy()
    beta2 = 0.5
    alpha2 = 2.0
    input_dict2 = {
        "input": input2,
        "vec1": vec12,
        "vec2": vec22,
        "beta": beta2,
        "alpha": alpha2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.ones(4, 5).numpy()
    vec13 = torch.arange(1, 5).float().numpy()
    vec23 = torch.arange(1, 6).float().numpy()
    beta3 = 0.0
    alpha3 = -1.0
    input_dict3 = {
        "input": input3,
        "vec1": vec13,
        "vec2": vec23,
        "beta": beta3,
        "alpha": alpha3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randint(-5, 5, (2, 2), dtype=torch.int32).numpy()
    vec14 = torch.tensor([-1, 2], dtype=torch.int32).numpy()
    vec24 = torch.tensor([3, -4], dtype=torch.int32).numpy()
    beta4 = 1
    alpha4 = 1
    input_dict4 = {
        "input": input4,
        "vec1": vec14,
        "vec2": vec24,
        "beta": beta4,
        "alpha": alpha4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randn(5, 1).numpy()
    vec15 = torch.randn(5).numpy()
    vec25 = torch.randn(1).numpy()
    beta5 = 0.25
    alpha5 = 0.75
    input_dict5 = {
        "input": input5,
        "vec1": vec15,
        "vec2": vec25,
        "beta": beta5,
        "alpha": alpha5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = addr_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('addr', generated_inputs)
