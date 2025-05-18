
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def prelu_inputs():
    list_of_inputs = []

    input = torch.randn(1, 3, 5, 5).numpy()
    num_parameters = 1
    init = 0.25

    input_dict = {
        "input": input,
        "num_parameters": num_parameters,
        "init": init
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 4, 4, 4).numpy()
    num_parameters = 4
    init = 0.0

    input_dict = {
        "input": input,
        "num_parameters": num_parameters,
        "init": init
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 2, 6, 6).numpy()
    num_parameters = 2
    init = 0.5

    input_dict = {
        "input": input,
        "num_parameters": num_parameters,
        "init": init
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 1, 5, 5).numpy()
    num_parameters = 1
    init = -0.25

    input_dict = {
        "input": input,
        "num_parameters": num_parameters,
        "init": init
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 8, 10, 10).numpy()
    num_parameters = 8
    init = 1.0

    input_dict = {
        "input": input,
        "num_parameters": num_parameters,
        "init": init
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = prelu_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('PReLU_', list_of_inputs)
