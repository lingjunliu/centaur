
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def hypot_inputs():
    list_of_inputs = []

    input1 = torch.randn(3, 4).numpy()
    input2 = torch.randn(3, 4).numpy()
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.randn(5).numpy()
    input2 = torch.randn(5).numpy()
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.randn(2, 2, 2).numpy()
    input2 = torch.randn(2, 2, 2).numpy()
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input1 = torch.randn(1).numpy()
    input2 = torch.randn(1).numpy()
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.randn(4, 1).numpy()
    input2 = torch.randn(4, 1).numpy()
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs

list_of_inputs = hypot_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('hypot', list_of_inputs)
