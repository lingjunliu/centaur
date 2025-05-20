
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def fix_inputs():
    list_of_inputs = []

    input_float = np.array([1.2, 2.7, -3.1, -4.8]).astype(np.float32)
    input_dict = {"input": input_float}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_int = np.array([1, 2, -3, -4]).astype(np.int32)
    input_dict = {"input": input_int}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_2d = np.array([[1.2, 2.7], [-3.1, -4.8]]).astype(np.float64)
    input_dict = {"input": input_2d}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_3d = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {"input": input_3d}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_large = np.array([1000.2, 2000.7, -3000.1, -4000.8]).astype(np.float32)
    input_dict = {"input": input_large}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_zeros = np.array([0.0, 0.0, 0.0]).astype(np.float32)
    input_dict = {"input": input_zeros}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = fix_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('fix', generated_inputs)
