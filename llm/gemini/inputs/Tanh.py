
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def Tanh_inputs():
    generated_inputs = []

    # Input 1: Scalar
    input1 = np.array(0.5)
    generated_inputs.append({"input": input1})

    # Input 2: 1D array (vector)
    input2 = np.array([-1.0, 0.0, 1.0, 2.0])
    generated_inputs.append({"input": input2})

    # Input 3: 2D array (matrix)
    input3 = np.array([[-2.0, -1.0], [0.0, 1.0], [2.0, 3.0]])
    generated_inputs.append({"input": input3})

    # Input 4: 3D array
    input4 = np.random.rand(2, 3, 4)
    generated_inputs.append({"input": input4})

    # Input 5: Array with negative values and zeros
    input5 = np.array([[-1, 0, 1], [-2, 0, 2]])
    generated_inputs.append({"input": input5})
    
    return generated_inputs

generated_inputs = Tanh_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('Tanh', generated_inputs)
