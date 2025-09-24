
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def tensordot_inputs():
    list_of_inputs = []

    a = np.random.rand(3, 4, 5).astype(np.float32)
    b = np.random.rand(4, 5, 6).astype(np.float32)
    dims = ([1, 2], [0, 1])
    input_dict = {"a": a, "b": b, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.random.rand(2, 3).astype(np.int32)
    b = np.random.rand(3, 4).astype(np.int32)
    dims = ([1], [0])
    input_dict = {"a": a, "b": b, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.random.rand(2, 3, 4).astype(np.float64)
    b = np.random.rand(4, 5).astype(np.float64)
    dims = ([2], [0])
    input_dict = {"a": a, "b": b, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.random.rand(2, 3).astype(np.complex64)
    b = np.random.rand(3, 4).astype(np.complex64)
    dims = ([1], [0])
    input_dict = {"a": a, "b": b, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.random.rand(5).astype(np.float32)
    b = np.random.rand(5).astype(np.float32)
    dims = ([0], [0])
    input_dict = {"a": a, "b": b, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.random.rand(2, 2, 2).astype(np.float32)
    b = np.random.rand(2, 2, 2).astype(np.float32)
    dims = ([0,1,2], [0,1,2])
    input_dict = {"a": a, "b": b, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.random.rand(2, 3, 4, 5).astype(np.float32)
    b = np.random.rand(5, 6, 7).astype(np.float32)
    dims = ([3], [0])
    input_dict = {"a": a, "b": b, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.random.rand(1).astype(np.float32)
    b = np.random.rand(1).astype(np.float32)
    dims = ([0], [0])
    input_dict = {"a": a, "b": b, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array([[1,2],[3,4]]).astype(np.float32)
    b = np.array([[5,6],[7,8]]).astype(np.float32)
    dims = (([0, 1]), ([0, 1]))
    input_dict = {"a": a, "b": b, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = tensordot_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('tensordot', generated_inputs)
