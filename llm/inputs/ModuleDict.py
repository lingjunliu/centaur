
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import torch.nn as nn
import copy

def ModuleDict_inputs():
    list_of_inputs = []

    # Input 1: Empty ModuleDict
    input_dict = {
        'modules': {}
    }
    list_of_inputs.append(input_dict)
    
    return list_of_inputs

generated_inputs = ModuleDict_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('ModuleDict', generated_inputs)
