
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def prod_inputs():
    list_of_inputs = []

    # Input 1: 2D float tensor, dim=0
    input1 = np.random.randn(4, 2).astype(np.float32)
    input_dict1 = {"input": input1, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D int tensor, dim=1
    input2 = np.random.randint(-5, 5, size=(3, 4)).astype(np.int32)
    input_dict2 = {"input": input2, "dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D float tensor, dim=2
    input3 = np.random.randn(2, 3, 5).astype(np.float64)
    input_dict3 = {"input": input3, "dim": 2}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 1D float tensor, dim=0
    input4 = np.random.randn(6).astype(np.float32)
    input_dict4 = {"input": input4, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 2D tensor with negative values, dim=0
    input5 = np.array([[-1.0, 2.0], [-3.0, 4.0]]).astype(np.float32)
    input_dict5 = {"input": input5, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 3D int tensor, dim=0
    input6 = np.random.randint(-5, 5, size=(2, 3, 4)).astype(np.int64)
    input_dict6 = {"input": input6, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: 2D tensor with large values, dim=1
    input7 = np.array([[1000, 2000], [3000, 4000]]).astype(np.float32)
    input_dict7 = {"input": input7, "dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = prod_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('prod', generated_inputs)
