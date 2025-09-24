
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def circularpad1d_inputs():
    list_of_inputs = []

    padding = 2
    input_dict = {
        "padding": padding,
        "input": np.arange(4, dtype=np.float32).reshape(2, 2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    padding = (3, 1)
    input_dict = {
        "padding": padding,
        "input": np.arange(6, dtype=np.float32).reshape(2, 3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = 0
    input_dict = {
        "padding": padding,
        "input": np.arange(4, dtype=np.float32).reshape(1, 4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    padding = (0, 0)
    input_dict = {
        "padding": padding,
        "input": np.arange(4, dtype=np.int32).reshape(1, 4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    padding = 1
    input_dict = {
        "padding": padding,
        "input": np.arange(4, dtype=np.float64).reshape(1, 4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = (1, 1)
    input_dict = {
        "padding": padding,
        "input": np.arange(4, dtype=np.int64).reshape(1, 4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    padding = 2
    input_dict = {
        "padding": padding,
        "input": np.arange(6, dtype=np.float32).reshape(2, 3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = circularpad1d_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('CircularPad1d', generated_inputs)
