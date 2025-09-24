
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def threshold_inputs():
    list_of_inputs = []

    input_1 = np.array([-1, 0, 1, 2], dtype=np.float32)
    threshold_1 = 1.0
    value_1 = 0.0
    input_dict_1 = {"input": input_1, "threshold": threshold_1, "value": value_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    input_2 = np.array([[-1, 0, 1], [2, 3, 4]], dtype=np.float64)
    threshold_2 = 2.0
    value_2 = -1.0
    input_dict_2 = {"input": input_2, "threshold": threshold_2, "value": value_2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    input_3 = np.array([[-1, 0, 1], [2, 3, 4]], dtype=np.int64)
    threshold_3 = 2
    value_3 = -1
    input_dict_3 = {"input": input_3, "threshold": float(threshold_3), "value": float(value_3)}
    list_of_inputs.append(copy.deepcopy(input_dict_3))
    
    input_4 = np.array([[-1.5, 0.5, 1.5], [2.5, 3.5, 4.5]], dtype=np.float32)
    threshold_4 = 2.0
    value_4 = 0.0
    input_dict_4 = {"input": input_4, "threshold": threshold_4, "value": value_4}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    input_5 = np.array([[-1, 0, 1], [2, 3, 4]], dtype=np.int32)
    threshold_5 = 1
    value_5 = 5
    input_dict_5 = {"input": input_5, "threshold": float(threshold_5), "value": float(value_5)}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    input_6 = np.array([[[1,2],[3,4]],[[5,6],[7,8]]], dtype=np.float32)
    threshold_6 = 4.0
    value_6 = 10.0
    input_dict_6 = {"input": input_6, "threshold": threshold_6, "value": value_6}
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    input_7 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    threshold_7 = 3.0
    value_7 = -2.0
    input_dict_7 = {"input": input_7, "threshold": threshold_7, "value": value_7}
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    return list_of_inputs

generated_inputs = threshold_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('threshold', generated_inputs)
