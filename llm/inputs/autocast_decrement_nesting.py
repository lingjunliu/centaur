
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def autocast_decrement_nesting_inputs():
    list_of_inputs = []
    list_of_inputs.append({})
    return list_of_inputs

generated_inputs = autocast_decrement_nesting_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('autocast_decrement_nesting', generated_inputs)
