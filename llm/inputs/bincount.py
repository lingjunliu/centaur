
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def bincount_inputs():
    list_of_inputs = []

    input = np.array([1, 2, 3, 4, 4, 1]).astype(np.int64)
    weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).astype(np.float32)
    minlength = 7

    input_dict = {
        "input": input,
        "weights": weights,
        "minlength": minlength
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.array([0, 1, 1, 3, 2, 1, 7]).astype(np.int64)
    weights = np.array([0.5, 0.2, 0.1, 0.8, 0.3, 0.9, 0.4]).astype(np.float32)
    minlength = 10
    input_dict = {
        "input": input,
        "weights": weights,
        "minlength": minlength
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.array([0, 0, 1, 2, 3, 3, 3]).astype(np.int64)
    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).astype(np.float32)
    minlength = 5
    input_dict = {
        "input": input,
        "weights": weights,
        "minlength": minlength
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.array([2, 2, 1, 1, 0, 1, 2]).astype(np.int64)
    weights = np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]).astype(np.float32)
    minlength = 4

    input_dict = {
        "input": input,
        "weights": weights,
        "minlength": minlength
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.array([5, 4, 3, 2, 1, 0, 0]).astype(np.int64)
    weights = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]).astype(np.float32)
    minlength = 8
    input_dict = {
        "input": input,
        "weights": weights,
        "minlength": minlength
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = bincount_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('bincount', list_of_inputs)
