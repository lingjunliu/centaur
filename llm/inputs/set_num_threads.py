
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def set_num_threads_inputs():
    list_of_inputs = []

    input_dict = {
        "num_threads": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_threads": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_threads": 4
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_threads": 8
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_threads": 16
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = set_num_threads_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('set_num_threads', generated_inputs)
