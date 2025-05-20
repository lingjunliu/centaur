
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def unravel_index_inputs():
    list_of_inputs = []

    indices = np.array([22, 41, 37]).astype(np.int64)
    shape = (7, 6)
    input_dict = {"indices": indices, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]).astype(np.int64)
    shape = (3, 4)
    input_dict = {"indices": indices, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = np.array([[0, 1], [2, 3], [4, 5]]).astype(np.int64)
    shape = (2, 3)
    input_dict = {"indices": indices, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]).astype(np.int64)
    shape = (2, 2)
    input_dict = {"indices": indices, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = np.array(7).astype(np.int64)
    shape = (3, 5)
    input_dict = {"indices": indices, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    indices = np.array([1, 2, 3, 4, 5, 6, 7, 8]).astype(np.int64)
    shape = (2, 2, 2)
    input_dict = {"indices": indices, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = np.array([1, 2, 3, 4]).astype(np.int64)
    shape = (2, 2)
    input_dict = {"indices": indices, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = unravel_index_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('unravel_index', generated_inputs)
