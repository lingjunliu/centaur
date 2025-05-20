
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def ZeroPad2d_inputs():
    list_of_inputs = []

    input_dict_1 = {
        "padding": 2,
        "input": np.random.randn(1, 1, 3, 3).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    input_dict_2 = {
        "padding": (1, 1, 2, 0),
        "input": np.random.randn(1, 1, 3, 3).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    input_dict_3 = {
        "padding": (0, 0, 0, 0),
        "input": np.random.randn(1, 1, 3, 3).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    input_dict_4 = {
        "padding": (1, 2, 3, 4),
        "input": np.random.randn(1, 1, 3, 3).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    
    input_dict_5 = {
        "padding": (5, 4, 3, 2),
        "input": np.random.randn(1, 1, 3, 3).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    return list_of_inputs

generated_inputs = ZeroPad2d_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('ZeroPad2d', generated_inputs)
