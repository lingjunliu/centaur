
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def interpolate_inputs():
    list_of_inputs = []

    input1 = torch.randn(1, 3, 10, 10).numpy()
    input_dict1 = {"input": input1, "size": (12, 12), "mode": "nearest"}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 4, 5, 5, 5).numpy()
    input_dict2 = {"input": input2, "scale_factor": 2, "mode": "trilinear"}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(1, 1, 15).numpy()
    input_dict3 = {"input": input3, "size": (20,), "mode": "linear"}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(1, 2, 8, 8).numpy()
    input_dict4 = {"input": input4, "scale_factor": 0.5, "mode": "bilinear"}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randn(1, 3, 7, 7).numpy()
    input_dict5 = {"input": input5, "size": (9, 9), "mode": "bicubic"}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = interpolate_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('interpolate', generated_inputs)
