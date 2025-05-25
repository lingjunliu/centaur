
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def ShortStorage_inputs():
    list_of_inputs = []

    input1 = {"size": 0, "input": np.array([])}
    list_of_inputs.append(copy.deepcopy(input1))

    input2 = {"size": 1, "input": np.array([1], dtype=np.int16)}
    list_of_inputs.append(copy.deepcopy(input2))

    input3 = {"size": 10, "input": np.arange(10, dtype=np.int16)}
    list_of_inputs.append(copy.deepcopy(input3))
    
    input4 = {"size": 100, "input": np.random.randint(-100, 100, size=100, dtype=np.int16)}
    list_of_inputs.append(copy.deepcopy(input4))

    input5 = {"size": 5, "input": np.array([-1, -2, 0, 1, 2], dtype=np.int16)}
    list_of_inputs.append(copy.deepcopy(input5))
    
    return list_of_inputs

generated_inputs = ShortStorage_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('ShortStorage', generated_inputs)
