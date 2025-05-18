
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def inner_inputs():
    list_of_inputs = []

    input1 = np.random.randn(3).astype(np.float32)
    input2 = np.random.randn(3).astype(np.float32)
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.random.randn(2, 3).astype(np.float32)
    input2 = np.random.randn(2, 3).astype(np.float32)
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.random.randn(2, 3, 4).astype(np.float32)
    input2 = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.random.randn(1, 5).astype(np.float32)
    input2 = np.random.randn(5).astype(np.float32)
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.random.randn(5).astype(np.float32)
    input2 = np.random.randn(1, 5).astype(np.float32)
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

list_of_inputs = inner_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('inner', list_of_inputs)
