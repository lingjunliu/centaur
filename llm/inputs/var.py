
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def var_inputs():
    list_of_inputs = []

    input_np = np.array([1.0, 2.0, 3.0, 4.0]).astype(np.float32)
    input_dict = {
        "input": input_np,
        "dim": 0,
        "correction": 1,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_np = np.array([[1.0, 2.0], [3.0, 4.0]]).astype(np.float64)
    input_dict = {
        "input": input_np,
        "dim": 1,
        "correction": 1,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_np = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).astype(np.float32)
    input_dict = {
        "input": input_np,
        "dim": 2,
        "correction": 0,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_np = np.array([[-1.0, 2.0], [3.0, -4.0]]).astype(np.float32)
    input_dict = {
        "input": input_np,
        "dim": 1,
        "correction": 1,
        "keepdim": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_np = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict = {
        "input": input_np,
        "dim": 1,
        "correction": 1,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_np = np.array([[1.0 + 1j, 2.0 - 2j], [3.0 + 3j, 4.0 - 4j]]).astype(np.complex64)
    input_dict = {
        "input": input_np,
        "dim": 1,
        "correction": 1,
        "keepdim": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = var_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('var', generated_inputs)
