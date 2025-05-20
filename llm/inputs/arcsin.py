
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def arcsin_inputs():
    generated_inputs = []

    # Input 1: Basic float tensor
    input1 = np.array([0.0, 0.5, -0.5])
    input_dict1 = {"input": input1}
    generated_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Multi-dimensional float tensor
    input2 = np.array([[0.2, 0.4], [-0.1, -0.3]])
    input_dict2 = {"input": input2}
    generated_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Single element float tensor
    input3 = np.array(0.7)
    input_dict3 = {"input": input3}
    generated_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Larger multi-dimensional tensor
    input4 = np.random.uniform(low=-1.0, high=1.0, size=(2, 3, 4))
    input_dict4 = {"input": input4}
    generated_inputs.append(copy.deepcopy(input_dict4))

    return generated_inputs

generated_inputs = arcsin_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('arcsin', generated_inputs)
