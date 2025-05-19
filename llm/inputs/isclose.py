
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def isclose_inputs():
    list_of_inputs = []

    input1 = np.array([1.0, 2.0, 3.0])
    other1 = np.array([1.0, 2.0, 3.1])
    input_dict1 = {
        "input": input1,
        "other": other1,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([1.0, np.nan, 3.0])
    other2 = np.array([1.0, np.nan, 3.0])
    input_dict2 = {
        "input": input2,
        "other": other2,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([1, 2, 3], dtype=np.int32)
    other3 = np.array([1, 2, 3], dtype=np.int32)
    input_dict3 = {
        "input": input3,
        "other": other3,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[1.0, 2.0], [3.0, 4.0]])
    other4 = np.array([[1.0, 2.1], [3.0, 4.0]])
    input_dict4 = {
        "input": input4,
        "other": other4,
        "rtol": 0.1,
        "atol": 0.0,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([-1.0, -2.0, -3.0])
    other5 = np.array([-1.0, -2.0, -3.1])
    input_dict5 = {
        "input": input5,
        "other": other5,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([1.0 + 1j, 2.0 + 2j])
    other6 = np.array([1.0 + 1j, 2.0 + 2.1j])
    input_dict6 = {
        "input": input6,
        "other": other6,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    other7 = np.array([[[1.0, 2.0], [3.1, 4.0]], [[5.0, 6.0], [7.0, 8.1]]])
    input_dict7 = {
        "input": input7,
        "other": other7,
        "rtol": 0.05,
        "atol": 0.0,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = isclose_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('isclose', generated_inputs)
