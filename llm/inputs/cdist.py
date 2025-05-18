
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def cdist_inputs():
    list_of_inputs = []

    x1 = torch.randn(5, 3).numpy()
    x2 = torch.randn(7, 3).numpy()
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = torch.randn(10, 2).numpy()
    x2 = torch.randn(5, 2).numpy()
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = torch.randn(2, 4, 3).numpy()
    x2 = torch.randn(2, 5, 3).numpy()
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = torch.randn(1, 8).numpy()
    x2 = torch.randn(1, 8).numpy()
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = torch.randn(3, 1, 5).numpy()
    x2 = torch.randn(3, 1, 5).numpy()
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = cdist_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('cdist', list_of_inputs)
