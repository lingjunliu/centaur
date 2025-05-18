
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def rand_like_inputs():
    list_of_inputs = []

    input1 = np.random.randn(3, 4).astype(np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.randint(0, 10, size=(2, 2, 2)).astype(np.int64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[1.0, 2.0], [3.0, 4.0]]).astype(np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.zeros(5).astype(np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.ones((2, 3, 4)).astype(np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('rand_like', list_of_inputs)
