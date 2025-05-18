
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def sub_inputs():
    list_of_inputs = []
    
    input = torch.randn(2, 3, 4, 5).numpy()
    other = torch.randn(2, 3, 4, 5).numpy()
    alpha = 1.0
    
    input_dict = {
        "input": input,
        "other": other,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 5, 5).numpy()
    other = torch.randn(1, 5, 5).numpy()
    alpha = 0.5
    
    input_dict = {
        "input": input,
        "other": other,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(3, 2).numpy()
    other = torch.randn(3, 2).numpy()
    alpha = 2.0
    
    input_dict = {
        "input": input,
        "other": other,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(4).numpy()
    other = torch.randn(4).numpy()
    alpha = 0.0
    
    input_dict = {
        "input": input,
        "other": other,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 1, 1).numpy()
    other = torch.randn(1, 1, 1).numpy()
    alpha = -1.0
    
    input_dict = {
        "input": input,
        "other": other,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

list_of_inputs = sub_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('sub', list_of_inputs)
