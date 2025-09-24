
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def argwhere_inputs():
    list_of_inputs = []

    # Input 1: 1D integer tensor
    input1 = np.array([1, 0, 2, 0, 3])
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D float tensor
    input2 = np.array([[1.0, 0.0, 2.5], [0.0, -1.0, 3.2]])
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D integer tensor with negative values
    input3 = np.array([[[1, 0, -1], [0, 2, 0]], [[-2, 0, 3], [0, -1, 0]]])
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 1D boolean tensor
    input4 = np.array([True, False, True, False, True])
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 2D complex tensor
    input5 = np.array([[1+1j, 0, 2-2j], [0, -1+0j, 3+1j]])
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: Empty array
    input6 = np.array([])
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: All zeros
    input7 = np.zeros((2,3))
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    # Input 8: 4D array
    input8 = np.random.randint(-5, 5, size=(2, 2, 2, 2))
    input_dict8 = {"input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    return list_of_inputs

generated_inputs = argwhere_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('argwhere', generated_inputs)
