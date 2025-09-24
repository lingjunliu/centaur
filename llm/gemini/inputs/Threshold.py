
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def threshold_inputs():
    list_of_inputs = []

    input_dict = {
        "input": np.array([1.0, 0.0, -1.0]),
        "threshold": 0.5,
        "value": 1.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([-0.5, -0.1, 0.2]),
        "threshold": -0.2,
        "value": 0.0,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([0.0, -0.5, 0.5]),
        "threshold": 0.0,
        "value": -1.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([2.0, 0.5, -0.5]),
        "threshold": 1.0,
        "value": 10.0,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([-2.0, -0.5, 0.5]),
        "threshold": -1.0,
        "value": -5.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = threshold_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('Threshold', generated_inputs)
