
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def bessel_j1_inputs():
    list_of_inputs = []

    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict1 = {"x": x1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    x2 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    input_dict2 = {"x": x2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    x3 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    input_dict3 = {"x": x3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    return list_of_inputs

generated_inputs = bessel_j1_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('bessel_j1', generated_inputs)
