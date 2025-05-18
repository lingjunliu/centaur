
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def broadcast_to_inputs():
    list_of_inputs = []

    # Example 1
    input = np.array([1, 2, 3])
    shape = [3, 3]
    input_dict = {"input": input, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2
    input = np.array([[1], [2]])
    shape = [2, 3]
    input_dict = {"input": input, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3
    input = np.array([1])
    shape = [2, 3, 4]
    input_dict = {"input": input, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4
    input = np.array([[1, 2], [3, 4]])
    shape = [2, 2, 2]
    input_dict = {"input": input, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5
    input = np.array([[[1], [2]], [[3], [4]]])
    shape = [2, 2, 3]
    input_dict = {"input": input, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = broadcast_to_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('broadcast_to', list_of_inputs)
