
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def local_response_norm_inputs():
    list_of_inputs = []

    size = 3
    alpha = 0.0001
    beta = 0.75
    k = 1.0
    input_data = torch.randn(1, 3, 24, 24).numpy()
    list_of_inputs.append({"size": size, "alpha": alpha, "beta": beta, "k": k, "input": input_data})

    size = 5
    alpha = 0.0002
    beta = 0.5
    k = 2.0
    input_data = torch.randn(1, 5, 12, 12).numpy()
    list_of_inputs.append({"size": size, "alpha": alpha, "beta": beta, "k": k, "input": input_data})

    size = 1
    alpha = 0.00005
    beta = 0.9
    k = 0.5
    input_data = torch.randn(1, 1, 32, 32).numpy()
    list_of_inputs.append({"size": size, "alpha": alpha, "beta": beta, "k": k, "input": input_data})

    size = 7
    alpha = 0.00015
    beta = 0.6
    k = 1.5
    input_data = torch.randn(1, 7, 8, 8).numpy()
    list_of_inputs.append({"size": size, "alpha": alpha, "beta": beta, "k": k, "input": input_data})

    size = 9
    alpha = 0.00025
    beta = 0.4
    k = 2.5
    input_data = torch.randn(1, 9, 16, 16).numpy()
    list_of_inputs.append({"size": size, "alpha": alpha, "beta": beta, "k": k, "input": input_data})

    return list_of_inputs

generated_inputs = local_response_norm_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('LocalResponseNorm', generated_inputs)
