
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def addr_inputs():
    list_of_inputs = []

    input_tensor = torch.randn(3, 3).numpy()
    vec1 = torch.randn(3).numpy()
    vec2 = torch.randn(3).numpy()
    beta = 1.0
    alpha = 1.0
    
    input_dict = {
        "input": input_tensor,
        "vec1": vec1,
        "vec2": vec2,
        "beta": beta,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(2, 4).numpy()
    vec1 = torch.randn(2).numpy()
    vec2 = torch.randn(4).numpy()
    beta = 0.5
    alpha = 2.0
    
    input_dict = {
        "input": input_tensor,
        "vec1": vec1,
        "vec2": vec2,
        "beta": beta,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(5, 5).numpy()
    vec1 = torch.randn(5).numpy()
    vec2 = torch.randn(5).numpy()
    beta = 0.0
    alpha = -1.0
    
    input_dict = {
        "input": input_tensor,
        "vec1": vec1,
        "vec2": vec2,
        "beta": beta,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(4, 2).numpy()
    vec1 = torch.randn(4).numpy()
    vec2 = torch.randn(2).numpy()
    beta = -0.5
    alpha = 0.5
    
    input_dict = {
        "input": input_tensor,
        "vec1": vec1,
        "vec2": vec2,
        "beta": beta,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(1, 1).numpy()
    vec1 = torch.randn(1).numpy()
    vec2 = torch.randn(1).numpy()
    beta = 2.0
    alpha = 3.0
    
    input_dict = {
        "input": input_tensor,
        "vec1": vec1,
        "vec2": vec2,
        "beta": beta,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

list_of_inputs = addr_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('addr', list_of_inputs)
