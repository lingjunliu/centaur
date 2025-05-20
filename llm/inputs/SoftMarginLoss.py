
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def soft_margin_loss_inputs():
    list_of_inputs = []

    input_dict1 = {
        'input': np.array([0.5, -0.2, 0.8]),
        'target': np.array([1, -1, 1]),
        'reduction': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input_dict2 = {
        'input': np.array([[0.5, -0.2], [0.8, -0.1]]),
        'target': np.array([[1, -1], [1, -1]]),
        'reduction': 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input_dict3 = {
        'input': np.array([0.2, -0.9, 0.5]),
        'target': np.array([-1, 1, -1]),
        'reduction': 'none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input_dict4 = {
        'input': np.array([1.0, -0.5, 0.0]),
        'target': np.array([1, -1, 1]),
        'reduction': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input_dict5 = {
        'input': np.array([-0.3, 0.7, -0.1]),
        'target': np.array([-1, 1, -1]),
        'reduction': 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = soft_margin_loss_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('SoftMarginLoss', generated_inputs)
