
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def matrix_power_inputs():
    list_of_inputs = []

    # Input 1: Basic float matrix
    input1 = np.random.rand(3, 3).astype(np.float32)
    n1 = 2
    input_dict1 = {"input": input1, "n": n1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Integer matrix
    input2 = np.random.randint(1, 5, size=(2, 2)).astype(np.int32)
    n2 = 3
    input_dict2 = {"input": input2, "n": n2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Complex matrix
    input3 = np.random.rand(2, 2) + 1j * np.random.rand(2, 2)
    input3 = input3.astype(np.complex64)
    n3 = 2
    input_dict3 = {"input": input3, "n": n3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Larger matrix
    input4 = np.random.rand(5, 5).astype(np.float64)
    n4 = 4
    input_dict4 = {"input": input4, "n": n4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Negative power
    input5 = np.random.rand(2, 2).astype(np.float32)
    n5 = -2
    input_dict5 = {"input": input5, "n": n5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Identity matrix
    input6 = np.eye(4).astype(np.float32)
    n6 = 5
    input_dict6 = {"input": input6, "n": n6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Small matrix with negative values
    input7 = np.random.randn(3, 3).astype(np.float32)
    n7 = 2
    input_dict7 = {"input": input7, "n": n7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs = matrix_power_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('matrix_power', generated_inputs)
