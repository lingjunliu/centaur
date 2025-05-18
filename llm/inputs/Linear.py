
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def linear_inputs():
    list_of_inputs = []

    input = torch.randn(1, 5).numpy()
    in_features = 5
    out_features = 3
    bias = True
    
    input_dict = {
        "input": input,
        "in_features": in_features,
        "out_features": out_features,
        "bias": bias
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 10).numpy()
    in_features = 10
    out_features = 7
    bias = False
    
    input_dict = {
        "input": input,
        "in_features": in_features,
        "out_features": out_features,
        "bias": bias
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 1).numpy()
    in_features = 1
    out_features = 1
    bias = True
    
    input_dict = {
        "input": input,
        "in_features": in_features,
        "out_features": out_features,
        "bias": bias
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(4, 20).numpy()
    in_features = 20
    out_features = 15
    bias = False
    
    input_dict = {
        "input": input,
        "in_features": in_features,
        "out_features": out_features,
        "bias": bias
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(3, 8).numpy()
    in_features = 8
    out_features = 12
    bias = True
    
    input_dict = {
        "input": input,
        "in_features": in_features,
        "out_features": out_features,
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

check_valid('Linear', list_of_inputs)
