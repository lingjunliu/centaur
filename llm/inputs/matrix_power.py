
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def matrix_power_inputs():
    list_of_inputs = []

    # Input 1: Basic float matrix
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    n1 = 2
    input_dict1 = {"input": input1, "n": n1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Integer matrix
    input2 = np.array([[1, 2], [3, 4]])
    n2 = 3
    input_dict2 = {"input": input2, "n": n2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Negative power
    input3 = np.array([[0.5, 0.2], [0.7, 0.9]])
    n3 = -1
    input_dict3 = {"input": input3, "n": n3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Larger matrix
    input4 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    n4 = 2
    input_dict4 = {"input": input4, "n": n4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Identity matrix
    input5 = np.eye(3)
    n5 = 5
    input_dict5 = {"input": input5, "n": n5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: Complex Matrix
    input6 = np.array([[1+1j, 2+2j], [3+3j, 4+4j]])
    n6 = 2
    input_dict6 = {"input": input6, "n": n6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Matrix with zero values
    input7 = np.array([[0, 1], [1, 0]])
    n7 = 4
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
