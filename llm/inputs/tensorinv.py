
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def torch_linalg_tensorinv_inputs():
    list_of_inputs = []

    # Example 1: Basic example from documentation
    A = torch.eye(4 * 6).reshape((4, 6, 8, 3)).numpy()
    ind = 2
    input_dict = {"a": A, "ind": ind}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = torch_linalg_tensorinv_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('tensorinv', generated_inputs)
