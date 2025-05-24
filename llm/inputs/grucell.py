
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def GRUCell_inputs():
    list_of_inputs = []

    input1 = torch.randn(3, 10).numpy()
    hidden1 = torch.randn(3, 20).numpy()
    bias1 = True
    input_dict1 = {
        "input": input1,
        "hidden": hidden1,
        "bias": bias1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(5, 15).numpy()
    hidden2 = torch.randn(5, 25).numpy()
    bias2 = False
    input_dict2 = {
        "input": input2,
        "hidden": hidden2,
        "bias": bias2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(7, 5).numpy()
    hidden3 = torch.randn(7, 10).numpy()
    bias3 = True
    input_dict3 = {
        "input": input3,
        "hidden": hidden3,
        "bias": bias3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(1, 8).numpy()
    hidden4 = torch.randn(1, 16).numpy()
    bias4 = False
    input_dict4 = {
        "input": input4,
        "hidden": hidden4,
        "bias": bias4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randn(4, 12).numpy()
    hidden5 = torch.randn(4, 24).numpy()
    bias5 = True
    input_dict5 = {
        "input": input5,
        "hidden": hidden5,
        "bias": bias5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(2, 7).numpy()
    hidden6 = torch.randn(2, 14).numpy()
    bias6 = False
    input_dict6 = {
        "input": input6,
        "hidden": hidden6,
        "bias": bias6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs = GRUCell_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('grucell', generated_inputs)
