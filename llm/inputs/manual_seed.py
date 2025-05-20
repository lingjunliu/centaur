
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def manual_seed_inputs():
    list_of_inputs = []

    seed1 = np.array(0)
    input_dict1 = {"seed": seed1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    seed2 = np.array(1)
    input_dict2 = {"seed": seed2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    seed3 = np.array(2147483647)
    input_dict3 = {"seed": seed3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    seed4 = np.array(-2147483648)
    input_dict4 = {"seed": seed4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    seed5 = np.array(12345)
    input_dict5 = {"seed": seed5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    seed6 = np.array(99999)
    input_dict6 = {"seed": seed6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs = manual_seed_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('manual_seed', generated_inputs)
