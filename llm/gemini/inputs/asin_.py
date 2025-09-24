
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def asin__inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor within [-1, 1]
    input1 = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D float tensor within [-1, 1]
    input2 = np.array([[0.2, 0.4, -0.1], [0.8, -0.9, 0.3]], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D float tensor within [-1, 1]
    input3 = np.random.uniform(low=-1.0, high=1.0, size=(2, 3, 4)).astype(np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Float tensor with a single value
    input4 = np.array([0.6], dtype=np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Larger tensor
    input5 = np.random.uniform(low=-1.0, high=1.0, size=(5, 5)).astype(np.float64)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: tensor with some edge values
    input6 = np.array([-1.0, -0.9, 0.0, 0.9, 1.0], dtype=np.float32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Large random tensor
    input7 = np.random.uniform(-1, 1, (10, 10, 10)).astype(np.float32)
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs = asin__inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('asin_', generated_inputs)
