
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def movedim_inputs():
    list_of_inputs = []

    input = torch.randn(2, 3, 4, 5).numpy()
    source = 1
    destination = 3
    input_dict = {
        "input": input,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 3, 4).numpy()
    source = 0
    destination = 2
    input_dict = {
        "input": input,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(5, 6).numpy()
    source = 0
    destination = 1
    input_dict = {
        "input": input,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(2, 3, 4, 5, 6).numpy()
    source = 4
    destination = 0
    input_dict = {
        "input": input,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(2, 3, 4).numpy()
    source = 2
    destination = 0
    input_dict = {
        "input": input,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = movedim_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('movedim', list_of_inputs)
