
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def dist_inputs():
    list_of_inputs = []

    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    other1 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    p1 = 2.0
    input_dict1 = {"input": input1, "other": other1, "p": p1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    other2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    p2 = 1.0
    input_dict2 = {"input": input2, "other": other2, "p": p2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    other3 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    p3 = 2.5
    input_dict3 = {"input": input3, "other": other3, "p": p3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input5 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    other5 = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float32)
    p5 = 0.5
    input_dict5 = {"input": input5, "other": other5, "p": p5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    other6 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    p6 = float('inf')
    input_dict6 = {"input": input6, "other": other6, "p": p6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape(2,2)
    other7 = np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32).reshape(2,2)
    p7 = 3.0
    input_dict7 = {"input": input7, "other": other7, "p": p7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = dist_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('dist', generated_inputs)
