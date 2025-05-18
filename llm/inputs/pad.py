
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def pad_inputs():
    list_of_inputs = []

    input = torch.randn(2, 3, 4, 5).numpy()
    pad = (1, 1, 2, 2, 0, 0, 0, 0)
    mode = 'constant'
    value = 0.0
    input_dict = {
        "input": input,
        "pad": pad,
        "mode": mode,
        "value": value
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 2, 3).numpy()
    pad = (2, 1, 0, 0, 0, 0)
    mode = 'constant'
    value = 0.0
    input_dict = {
        "input": input,
        "pad": pad,
        "mode": mode,
        "value": value
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(4, 5).numpy()
    pad = (0, 1, 2, 0)
    mode = 'constant'
    value = 0.0
    input_dict = {
        "input": input,
        "pad": pad,
        "mode": mode,
        "value": value
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 3, 32, 32).numpy()
    pad = (3, 3, 2, 2, 0, 0, 0, 0)
    mode = 'constant'
    value = 0.0
    input_dict = {
        "input": input,
        "pad": pad,
        "mode": mode,
        "value": value
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(1, 1, 5, 5).numpy()
    pad = (1, 2, 3, 4, 0, 0, 0, 0)
    mode = 'constant'
    value = 1.5
    input_dict = {
        "input": input,
        "pad": pad,
        "mode": mode,
        "value": value
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = pad_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('pad', list_of_inputs)
