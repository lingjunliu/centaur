
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def conj_physical_inputs():
    list_of_inputs = []

    # Input 1: Complex tensor
    input1 = np.array([1 + 1j, 2 + 2j, 3 + 3j])
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Multidimensional complex tensor
    input2 = np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]])
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Complex tensor with negative values
    input3 = np.array([-1 - 1j, -2 - 2j, -3 - 3j])
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Larger complex tensor
    input4 = np.array([
        [1 + 1j, 2 + 2j, 3 + 3j],
        [4 + 4j, 5 + 5j, 6 + 6j],
        [7 + 7j, 8 + 8j, 9 + 9j]
    ])
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Complex tensor with mixed positive and negative values
    input5 = np.array([1 - 1j, -2 + 2j, 3 - 3j, -4 + 4j])
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 3D complex tensor
    input6 = np.array([
        [[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]],
        [[5 + 5j, 6 + 6j], [7 + 7j, 8 + 8j]]
    ])
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Complex tensor with zero values
    input7 = np.array([0 + 0j, 1 + 1j, 0 - 1j, -1 + 0j])
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = conj_physical_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('conj_physical', generated_inputs)
