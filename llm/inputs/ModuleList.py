
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import torch.nn as nn
import copy
import numpy as np

def modulelist_inputs():
    list_of_inputs = []

    # Case 1: Empty ModuleList
    input_dict = {
        "modules": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: ModuleList with Linear layers
    input_dict = {
        "modules": [nn.Linear(10, 20), nn.Linear(20, 30)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: ModuleList with Conv2d and ReLU
    input_dict = {
        "modules": [nn.Conv2d(3, 16, kernel_size=3), nn.ReLU()]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: ModuleList with BatchNorm1d and Dropout
    input_dict = {
        "modules": [nn.BatchNorm1d(100), nn.Dropout(0.5)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 5: ModuleList with a mix of different module types
    input_dict = {
        "modules": [nn.Linear(5, 10), nn.ReLU(), nn.Conv1d(1, 3, kernel_size=3)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: ModuleList with Sequential
    input_dict = {
        "modules": [nn.Sequential(nn.Linear(10, 5), nn.ReLU())]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: ModuleList with nested ModuleList
    input_dict = {
        "modules": [nn.ModuleList([nn.Linear(5,5), nn.ReLU()]), nn.Linear(10, 10)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = modulelist_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('ModuleList', generated_inputs)
