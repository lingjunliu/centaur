
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def softmin_inputs():
    generated_inputs = []

    # Input 1: 1D tensor, dim=0
    input1 = np.array([1.0, 2.0, 3.0])
    input_dict1 = {"input": input1, "dim": 0}
    generated_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor, dim=0
    input2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    input_dict2 = {"input": input2, "dim": 0}
    generated_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 2D tensor, dim=1
    input3 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    input_dict3 = {"input": input3, "dim": 1}
    generated_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 3D tensor, dim=2
    input4 = np.random.rand(2, 3, 4)
    input_dict4 = {"input": input4, "dim": 2}
    generated_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 1D tensor with negative values, dim=0
    input5 = np.array([-1.0, -2.0, -3.0])
    input_dict5 = {"input": input5, "dim": 0}
    generated_inputs.append(copy.deepcopy(input_dict5))

    return generated_inputs

generated_inputs = softmin_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('Softmin', generated_inputs)
