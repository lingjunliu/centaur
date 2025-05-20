
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def gammainc_inputs():
    list_of_inputs = []

    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    input_dict = {"a": a, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    x = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64)
    input_dict = {"a": a, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([1, 2, 3], dtype=np.int32).astype(np.float32)
    x = np.array([4, 5, 6], dtype=np.int32).astype(np.float32)
    input_dict = {"a": a, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    x = np.array([1.1, 1.2, 1.3], dtype=np.float32)
    input_dict = {"a": a, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([1.0, 2.0, 3.0, 4.0]).reshape(2, 2).astype(np.float32)
    x = np.array([0.5, 1.5, 2.5, 3.5]).reshape(2, 2).astype(np.float32)
    input_dict = {"a": a, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = gammainc_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('gammainc', generated_inputs)
