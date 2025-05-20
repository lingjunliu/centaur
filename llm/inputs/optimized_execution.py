
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy

def optimized_execution_inputs():
    list_of_inputs = []

    input1 = {'enabled': True}
    list_of_inputs.append(input1)

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
