
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def bitwise_xor_inputs():
    list_of_inputs = []

    input1 = np.array([1, 2, 3, 4], dtype=np.int32)
    other1 = np.array([5, 6, 7, 8], dtype=np.int32)
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[1, 2], [3, 4]], dtype=np.int64)
    other2 = np.array([[5, 6], [7, 8]], dtype=np.int64)
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[-1, -2], [-3, -4]], dtype=np.int8)
    other3 = np.array([[5, 6], [7, 8]], dtype=np.int8)
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([1, 2, 3, 4], dtype=np.uint8)
    other4 = np.array([5, 6, 7, 8], dtype=np.uint8)
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int16)
    other5 = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.int16)
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([1, 0, 1, 0], dtype=bool)
    other6 = np.array([0, 1, 0, 1], dtype=bool)
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs = bitwise_xor_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('bitwise_xor', generated_inputs)
