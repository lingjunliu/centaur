
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def rfft2_inputs():
    list_of_inputs = []

    input1 = torch.randn(10, 10).numpy()
    input_dict1 = {
        "input": input1,
        "s": None,
        "dim": (-2, -1),
        "norm": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(5, 5, 5).numpy()
    input_dict2 = {
        "input": input2,
        "s": (10, 10),
        "dim": (-2, -1),
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(8, 8).numpy()
    input_dict3 = {
        "input": input3,
        "s": (4, 4),
        "dim": (0, 1),
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(12, 12).numpy()
    input_dict4 = {
        "input": input4,
        "s": (16, 16),
        "dim": (-2, -1),
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(4, 4, 4, 4).numpy()
    input_dict5 = {
        "input": input5,
        "s": (8, 8),
        "dim": (-2, -1),
        "norm": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(3, 5).numpy()
    input_dict6 = {
        "input": input6,
        "s": (6, 10),
        "dim": (0, 1),
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.randn(2, 7, 9).numpy()
    input_dict7 = {
        "input": input7,
        "s": (4, 10),
        "dim": (1, 2),
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = torch.randn(4, 6).numpy()
    input_dict8 = {
        "input": input8,
        "s": None,
        "dim": (0, 1),
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    return list_of_inputs

generated_inputs = rfft2_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('rfft2', generated_inputs)
