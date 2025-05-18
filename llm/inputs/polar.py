
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def polar_inputs():
    list_of_inputs = []

    abs_val = torch.randn(3, 4).numpy()
    angle_val = torch.randn(3, 4).numpy()
    input_dict = {
        "abs": abs_val,
        "angle": angle_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    abs_val = torch.randn(2, 2, 2).numpy()
    angle_val = torch.randn(2, 2, 2).numpy()
    input_dict = {
        "abs": abs_val,
        "angle": angle_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    abs_val = torch.randn(5).numpy()
    angle_val = torch.randn(5).numpy()
    input_dict = {
        "abs": abs_val,
        "angle": angle_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    abs_val = torch.randn(1, 1).numpy()
    angle_val = torch.randn(1, 1).numpy()
    input_dict = {
        "abs": abs_val,
        "angle": angle_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    abs_val = torch.randn(2, 3, 4, 5).numpy()
    angle_val = torch.randn(2, 3, 4, 5).numpy()
    input_dict = {
        "abs": abs_val,
        "angle": angle_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = polar_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('polar', list_of_inputs)
