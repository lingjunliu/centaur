
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def spherical_bessel_j0_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Negative values
    input2 = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Zero value
    input3 = np.array([0.0], dtype=np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Multi-dimensional array
    input4 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Large values
    input5 = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = spherical_bessel_j0_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('spherical_bessel_j0', generated_inputs)
