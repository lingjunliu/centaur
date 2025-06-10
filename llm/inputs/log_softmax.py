
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

generated_inputs = dict()

import torch, copy
import numpy as np

def log_softmax_inputs():
    list_of_inputs = []

    input1 = np.random.randn(3, 5).astype(np.float32)
    dim1 = 1
    dtype1 = None
    input_dict1 = {
        "input": input1,
        "dim": dim1,
        "dtype": dtype1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.randn(2, 4, 6).astype(np.float64)
    dim2 = 0
    dtype2 = None
    input_dict2 = {
        "input": input2,
        "dim": dim2,
        "dtype": dtype2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.randn(1, 3, 5, 7).astype(np.float16)
    dim3 = 2
    dtype3 = None
    input_dict3 = {
        "input": input3,
        "dim": dim3,
        "dtype": dtype3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.randn(4).astype(np.float32)
    dim4 = 0
    dtype4 = None
    input_dict4 = {
        "input": input4,
        "dim": dim4,
        "dtype": dtype4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.randn(2, 2).astype(np.float32)
    dim5 = -1
    dtype5 = None
    input_dict5 = {
        "input": input5,
        "dim": dim5,
        "dtype": dtype5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.random.randn(2, 2).astype(np.float32)
    dim6 = -2
    dtype6 = None
    input_dict6 = {
        "input": input6,
        "dim": dim6,
        "dtype": dtype6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.log_softmax"] = log_softmax_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('log_softmax', generated_inputs['torch.nn.functional.log_softmax'], lib="torch")
