
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def l1loss_inputs():
    list_of_inputs = []

    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    target1 = np.array([1.0, 2.5, 3.0], dtype=np.float32)
    input_dict1 = {"input": input1, "target": target1, "reduction": 'mean'}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    target2 = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)
    input_dict2 = {"input": input2, "target": target2, "reduction": 'mean'}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    target3 = np.array([[1.0, -2.0], [-3.0, 4.0]], dtype=np.float32)
    input_dict3 = {"input": input3, "target": target3, "reduction": 'mean'}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([1, 2, 3], dtype=np.float32)
    target4 = np.array([2, 3, 4], dtype=np.float32)
    input_dict4 = {"input": input4, "target": target4, "reduction": 'mean'}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    target5 = np.array([2.0, 3.0, 4.0], dtype=np.float64)
    input_dict5 = {"input": input5, "target": target5, "reduction": 'mean'}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = l1loss_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('L1Loss', generated_inputs)
