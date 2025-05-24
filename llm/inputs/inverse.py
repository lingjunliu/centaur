
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def inverse_inputs():
    list_of_inputs = []

    # Input 1: Basic square matrix (float)
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Larger square matrix (float)
    input2 = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Square matrix with negative values (float)
    input3 = np.array([[-1.0, 2.0], [3.0, -4.0]])
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Square matrix (double)
    input4 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 3D tensor of square matrices (float)
    input5 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 7.0], [6.0, 8.0]]])
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 7: Complex square matrix
    input7 = np.array([[1+1j, 2+2j], [3+3j, 4+4j]])
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs = inverse_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('inverse', generated_inputs)
