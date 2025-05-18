
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def gt_inputs():
    list_of_inputs = []

    input1 = np.array([[1, 2], [3, 4]])
    other1 = np.array([[0, 3], [3, 1]])
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([5, 6, 7])
    other2 = 6
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[0.1, 0.2], [0.3, 0.4]])
    other3 = np.array([[0.2, 0.1], [0.4, 0.3]])
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([-1, 0, 1])
    other4 = 0
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    other5 = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

list_of_inputs = gt_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('gt', list_of_inputs)
