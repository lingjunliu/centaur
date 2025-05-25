
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def finfo_inputs():
    list_of_inputs = []

    input1 = {'dtype': np.float16}
    list_of_inputs.append(copy.deepcopy(input1))

    input2 = {'dtype': np.float32}
    list_of_inputs.append(copy.deepcopy(input2))

    input3 = {'dtype': np.float64}
    list_of_inputs.append(copy.deepcopy(input3))

    input4 = {'dtype': np.complex64}
    list_of_inputs.append(copy.deepcopy(input4))

    input5 = {'dtype': np.complex128}
    list_of_inputs.append(copy.deepcopy(input5))

    return list_of_inputs

generated_inputs = finfo_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('finfo', generated_inputs)
