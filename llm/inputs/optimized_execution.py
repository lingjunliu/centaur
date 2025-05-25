
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def optimized_execution_inputs():
    list_of_inputs = []

    input1 = {'enabled': True}
    list_of_inputs.append(copy.deepcopy(input1))

    input2 = {'enabled': False}
    list_of_inputs.append(copy.deepcopy(input2))

    input3 = {'enabled': np.bool_(True).item()}
    list_of_inputs.append(copy.deepcopy(input3))

    input4 = {'enabled': np.bool_(False).item()}
    list_of_inputs.append(copy.deepcopy(input4))

    input5 = {'enabled': True}
    list_of_inputs.append(copy.deepcopy(input5))
    
    return list_of_inputs

generated_inputs = optimized_execution_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('optimized_execution', generated_inputs)
