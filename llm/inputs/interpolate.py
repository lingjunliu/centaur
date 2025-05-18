
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy

def interpolate_inputs():
    list_of_inputs = []

    input = torch.randn(1, 3, 10, 10).numpy()
    input_dict = {
        "input": input,
        "size": (12, 12)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 4, 5, 5).numpy()
    input_dict = {
        "input": input,
        "scale_factor": 2.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 1, 12, 12).numpy()
    input_dict = {
        "input": input,
        "size": (6, 6)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(4, 2, 8, 8).numpy()
    input_dict = {
        "input": input,
        "scale_factor": 0.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(3, 5, 15, 15).numpy()
    input_dict = {
        "input": input,
        "size": (20, 20)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = interpolate_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('interpolate', list_of_inputs)
