
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def vitals_enabled_inputs():
    list_of_inputs = []
    
    # Input is not actually used, but driver expects a dictionary with a valid 'input' key.
    input1 = {"input": np.array([1.0])}
    list_of_inputs.append(copy.deepcopy(input1))
    
    return list_of_inputs

generated_inputs = vitals_enabled_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('vitals_enabled', generated_inputs)
