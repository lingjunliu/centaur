
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def logdet_inputs():
    list_of_inputs = []

    A = np.random.rand(3, 3).astype(np.float32)
    input_dict = {"input": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = np.random.rand(2, 2).astype(np.float32)
    input_dict = {"input": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = np.array([[1.0, 0.0], [0.0, 1.0]]).astype(np.float32)
    input_dict = {"input": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = np.random.rand(3, 3).astype(np.float32)
    A = np.expand_dims(A, axis=0)
    input_dict = {"input": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = np.random.rand(2, 2).astype(np.float32)
    A = np.expand_dims(A, axis=0)
    input_dict = {"input": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = logdet_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('logdet', generated_inputs)
