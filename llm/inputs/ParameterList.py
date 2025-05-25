
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import torch.nn as nn
import numpy as np
import copy

def ParameterList_inputs():
    list_of_inputs = []

    # Input 1: Empty list
    input1 = []
    input_dict1 = {"parameter_list": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: List of float tensors
    input2 = [torch.randn(2, 3).detach().numpy(), torch.randn(5).detach().numpy()]
    input_dict2 = {"parameter_list": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: List of int tensors converted to float
    input3 = [torch.randint(0, 10, (2, 2)).float().numpy(), torch.randint(-5, 5, (3,)).float().numpy()]
    input_dict3 = {"parameter_list": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: List of Parameter objects
    input4 = [nn.Parameter(torch.randn(2, 2)).detach().numpy(), nn.Parameter(torch.randn(3)).detach().numpy()]
    input_dict4 = {"parameter_list": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: List of mixed tensors and parameters
    input5 = [torch.randn(2, 2).detach().numpy(), nn.Parameter(torch.randn(3)).detach().numpy()]
    input_dict5 = {"parameter_list": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = ParameterList_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('ParameterList', generated_inputs)
