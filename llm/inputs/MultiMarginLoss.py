
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def multimarginloss_inputs():
    list_of_inputs = []

    input = torch.randn(3, 5).numpy()
    target = torch.randint(1, 5, (3,)).numpy()
    input_dict = {
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(5, 10).numpy()
    target = torch.randint(1, 10, (5,)).numpy()
    input_dict = {
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(10, 3).numpy()
    target = torch.randint(1, 3, (10,)).numpy()
    input_dict = {
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(2, 4).numpy()
    target = torch.randint(1, 4, (2,)).numpy()
    input_dict = {
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(4, 2).numpy()
    target = torch.randint(1, 2, (4,)).numpy()
    input_dict = {
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = multimarginloss_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('MultiMarginLoss', list_of_inputs)
