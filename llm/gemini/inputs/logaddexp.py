
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def logaddexp_inputs():
    generated_inputs = []

    input1 = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    other1 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    input_dict1 = {"input": input1, "other": other1}
    generated_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([-100.0, -200.0, -300.0], dtype=np.float64)
    other2 = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    input_dict2 = {"input": input2, "other": other2}
    generated_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([1.0, 2000.0, 30000.0], dtype=np.float16)
    other3 = np.array([-1.0, -2.0, -3.0], dtype=np.float16)
    input_dict3 = {"input": input3, "other": other3}
    generated_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    other4 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    input_dict4 = {"input": input4, "other": other4}
    generated_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([1.0], dtype=np.float32)
    other5 = np.array([[-1.0, -2.0, -3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    input_dict5 = {"input": input5, "other": other5}
    generated_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[-1.0, -2.0, -3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    other6 = np.array([1.0], dtype=np.float32)
    input_dict6 = {"input": input6, "other": other6}
    generated_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[-1.0, -2.0], [-3.0, 4.0]], dtype=np.float32)
    other7 = np.array([[-1.0, -2.0], [-3.0, 4.0]], dtype=np.float32)
    input_dict7 = {"input": input7, "other": other7}
    generated_inputs.append(copy.deepcopy(input_dict7))

    return generated_inputs

generated_inputs = logaddexp_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('logaddexp', generated_inputs)
