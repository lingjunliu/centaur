
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def multiply_inputs():
    list_of_inputs = []

    input1 = np.random.randn(3, 4).astype(np.float32)
    other1 = np.random.randn(3, 4).astype(np.float32)
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.randint(-5, 5, size=(2, 2), dtype=np.int32)
    other2 = np.random.randint(-5, 5, size=(2, 2), dtype=np.int32)
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([1, 2, 3], dtype=np.float64)
    other3 = np.array([4, 5, 6], dtype=np.float64)
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.randn(1, 5, 5).astype(np.float32)
    other4 = np.random.randn(1, 5, 5).astype(np.float32)
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array(-2.5)
    other5 = np.array(3.0)
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([1+1j, 2+2j])
    other6 = np.array([3+3j, 4+4j])
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.array([[-1, -2], [-3, -4]], dtype=np.int8)
    other7 = np.array([[5, 6], [7, 8]], dtype=np.int8)
    input_dict7 = {"input": input7, "other": other7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = multiply_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('multiply', generated_inputs)
