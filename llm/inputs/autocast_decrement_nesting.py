
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def autocast_decrement_nesting_inputs():
    list_of_inputs = []

    input1 = {"input": np.array([1])}
    list_of_inputs.append(copy.deepcopy(input1))

    input2 = {"input": np.array([1.0])}
    list_of_inputs.append(copy.deepcopy(input2))

    input3 = {"input": np.array([1+1j])}
    list_of_inputs.append(copy.deepcopy(input3))
    
    input4 = {"input": np.array([[1,2],[3,4]])}
    list_of_inputs.append(copy.deepcopy(input4))

    input5 = {"input": np.array([[[1,2],[3,4]],[[5,6],[7,8]]])}
    list_of_inputs.append(copy.deepcopy(input5))

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
