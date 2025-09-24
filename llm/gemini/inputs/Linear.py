
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def linear_inputs():
    list_of_inputs = []

    # Input 1: Basic float input with bias
    input1 = torch.randn(3, 5).numpy()
    in_features1 = 5
    out_features1 = 4
    bias1 = True
    input_dict1 = {
        "input": input1,
        "in_features": in_features1,
        "out_features": out_features1,
        "bias": bias1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Integer input without bias
    input2 = torch.randint(0, 10, (2, 6)).float().numpy() # Convert to float
    in_features2 = 6
    out_features2 = 3
    bias2 = False
    input_dict2 = {
        "input": input2,
        "in_features": in_features2,
        "out_features": out_features2,
        "bias": bias2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    # Input 3: Input with negative values and bias
    input3 = torch.randn(4, 7) * -1.0
    input3 = input3.numpy()
    in_features3 = 7
    out_features3 = 2
    bias3 = True
    input_dict3 = {
        "input": input3,
        "in_features": in_features3,
        "out_features": out_features3,
        "bias": bias3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 1D input
    input4 = torch.randn(10).numpy()
    in_features4 = 10
    out_features4 = 5
    bias4 = True
    input_dict4 = {
        "input": input4,
        "in_features": in_features4,
        "out_features": out_features4,
        "bias": bias4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Higher dimensional input
    input5 = torch.randn(2, 3, 4).numpy()
    in_features5 = 4
    out_features5 = 2
    bias5 = False
    input_dict5 = {
        "input": input5,
        "in_features": in_features5,
        "out_features": out_features5,
        "bias": bias5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: Zero input, with bias
    input6 = torch.zeros(2, 3).numpy()
    in_features6 = 3
    out_features6 = 4
    bias6 = True
    input_dict6 = {
        "input": input6,
        "in_features": in_features6,
        "out_features": out_features6,
        "bias": bias6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Large in_features and out_features, with bias
    input7 = torch.randn(1, 100).numpy()
    in_features7 = 100
    out_features7 = 50
    bias7 = True
    input_dict7 = {
        "input": input7,
        "in_features": in_features7,
        "out_features": out_features7,
        "bias": bias7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = linear_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('Linear', generated_inputs)
