
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def cross_inputs():
    list_of_inputs = []

    input1 = np.array([1, 0, 0])
    other1 = np.array([0, 1, 0])
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([1, 2, 3])
    other2 = np.array([4, 5, 6])
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[1, 2, 3], [4, 5, 6]])
    other3 = np.array([[7, 8, 9], [10, 11, 12]])
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([1.5, 2.5, 3.5])
    other4 = np.array([4.5, 5.5, 6.5])
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([0, -1, 1])
    other5 = np.array([1, 1, 0])
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

list_of_inputs = cross_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('cross', list_of_inputs)
