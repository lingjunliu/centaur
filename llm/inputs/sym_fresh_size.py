
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def sym_fresh_size_inputs():
    list_of_inputs = []

    # Input 1: Simple string
    input1 = {"name": "x"}
    list_of_inputs.append(copy.deepcopy(input1))

    return list_of_inputs

generated_inputs = sym_fresh_size_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('sym_fresh_size', generated_inputs)
