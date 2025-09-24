
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def allclose_inputs():
    list_of_inputs = []

    # Case 1: Simple float tensors
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input2 = np.array([1.001, 2.002, 3.003], dtype=np.float32)
    input_dict = {
        "input": input1,
        "other": input2,
        "atol": 0.01,
        "rtol": 0.001,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Integer tensors
    input1 = np.array([1, 2, 3], dtype=np.int32)
    input2 = np.array([1, 2, 4], dtype=np.int32)
    input_dict = {
        "input": input1,
        "other": input2,
        "atol": 1,
        "rtol": 0.0,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Multidimensional tensors
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input2 = np.array([[1.0, 2.0], [3.0, 4.1]], dtype=np.float32)
    input_dict = {
        "input": input1,
        "other": input2,
        "atol": 0.0,
        "rtol": 0.01,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Negative values
    input1 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    input2 = np.array([-1.001, -2.002, -3.003], dtype=np.float32)
    input_dict = {
        "input": input1,
        "other": input2,
        "atol": 0.01,
        "rtol": 0.001,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 5: Tensors with NaN
    input1 = np.array([1.0, float('nan'), 3.0], dtype=np.float32)
    input2 = np.array([1.0, float('nan'), 3.0], dtype=np.float32)
    input_dict = {
        "input": input1,
        "other": input2,
        "atol": 0.0,
        "rtol": 0.0,
        "equal_nan": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = allclose_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('allclose', generated_inputs)
