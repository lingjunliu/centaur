
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def ndtri_inputs():
    list_of_inputs = []

    x1 = np.array(0.5, dtype=np.float32)
    input_dict1 = {"x": x1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    x2 = np.array([0.1, 0.5, 0.9], dtype=np.float64)
    input_dict2 = {"x": x2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    x3 = np.array([[0.2, 0.6], [0.4, 0.8]], dtype=np.float32)
    input_dict3 = {"x": x3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    x4 = np.array(0.99999, dtype=np.float64)
    input_dict4 = {"x": x4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    x5 = np.array([[[0.3, 0.7], [0.5, 0.9]], [[0.1, 0.4], [0.2, 0.6]]], dtype=np.float32)
    input_dict5 = {"x": x5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = ndtri_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('ndtri', generated_inputs)
