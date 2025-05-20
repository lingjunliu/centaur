
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def clamp_max_inputs():
    list_of_inputs = []

    input1 = np.array([-1, 0, 1, 2, 3], dtype=np.int32)
    max1 = 2.0
    input_dict1 = {"input": input1, "max": max1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([-1.5, 0.0, 1.5, 2.5, 3.5], dtype=np.float32)
    max2 = 2.0
    input_dict2 = {"input": input2, "max": max2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[-1, 0, 1], [2, 3, 4]], dtype=np.int64)
    max3 = 3.0
    input_dict3 = {"input": input3, "max": max3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[-1.5, 0.0, 1.5], [2.5, 3.5, 4.5]], dtype=np.float64)
    max4 = 3.5
    input_dict4 = {"input": input4, "max": max4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int16)
    max5 = 5.0
    input_dict5 = {"input": input5, "max": max5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[[1.1, 2.2], [3.3, 4.4]], [[5.5, 6.6], [7.7, 8.8]]], dtype=np.float16)
    max6 = 6.6
    input_dict6 = {"input": input6, "max": max6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([1, 2, 3], dtype=np.uint8)
    max7 = 2.0
    input_dict7 = {"input": input7, "max": max7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs = clamp_max_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('clamp_max', generated_inputs)
