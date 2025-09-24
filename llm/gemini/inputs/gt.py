
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def gt_inputs():
    list_of_inputs = []

    input1 = np.array([1, 2, 3, 4, 5])
    other1 = np.array([2, 2, 2, 2, 2])
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    other2 = np.array([[0.5, 2.5], [3.5, 3.5]])
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[-1, -2], [3, 4]])
    other3 = np.array([0])
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([1, 2, 3])
    other4 = 2
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    other5 = np.array([[[0, 3], [2, 5]], [[4, 7], [6, 9]]])
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([1, 2, 3], dtype=np.int64)
    other6 = np.array([0, 1, 4], dtype=np.int64)
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    other7 = np.array([0.5, 1.5, 4.5], dtype=np.float64)
    input_dict7 = {"input": input7, "other": other7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs = gt_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('gt', generated_inputs)
