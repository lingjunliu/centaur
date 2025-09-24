
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def trapz_inputs():
    list_of_inputs = []

    y = np.array([1, 2, 3], dtype=np.float32)
    x = np.array([4, 6, 8], dtype=np.float32)
    dx = 1.0
    dim = 0
    input_dict = {"y": y, "x": x, "dx": dx, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    x = np.array([[4, 6, 8], [10, 12, 14]], dtype=np.float32)
    dx = 1.0
    dim = 1
    input_dict = {"y": y, "x": x, "dx": dx, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = np.array([1, 2, 3], dtype=np.float32)
    dx = 2.0
    dim = 0
    input_dict = {"y": y, "x": np.array([0, 1, 2], dtype=np.float32), "dx": dx, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    dx = 0.5
    dim = 1
    input_dict = {"y": y, "x": np.array([[0, 1, 2], [0, 1, 2]], dtype=np.float32), "dx": dx, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    dx = 0.5
    dim = 0
    input_dict = {"y": y, "x": np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32), "dx": dx, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = trapz_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('trapz', generated_inputs)
