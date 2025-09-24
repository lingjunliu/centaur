
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def clip_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input1 = np.random.randn(3, 4).astype(np.float32)
    min_val1 = -1.0
    max_val1 = 1.0
    input_dict1 = {"input": input1, "min": min_val1, "max": max_val1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Integer tensor
    input2 = np.random.randint(-5, 5, size=(2, 2), dtype=np.int32)
    min_val2 = 0.0
    max_val2 = 3.0
    input_dict2 = {"input": input2, "min": min_val2, "max": max_val2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 1D tensor
    input3 = np.random.rand(5).astype(np.float64)
    min_val3 = 0.2
    max_val3 = 0.8
    input_dict3 = {"input": input3, "min": min_val3, "max": max_val3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 3D tensor with negative values
    input4 = np.random.randn(2, 3, 2).astype(np.float32) * 5 - 2.5
    min_val4 = -1.5
    max_val4 = 2.0
    input_dict4 = {"input": input4, "min": min_val4, "max": max_val4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Tensor with all same values
    input5 = np.full((4, 4), 2.0, dtype=np.float32)
    min_val5 = 1.0
    max_val5 = 3.0
    input_dict5 = {"input": input5, "min": min_val5, "max": max_val5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Mixed positive and negative boundaries
    input6 = np.random.randn(2, 2).astype(np.float32)
    min_val6 = -0.5
    max_val6 = 0.5
    input_dict6 = {"input": input6, "min": min_val6, "max": max_val6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Double tensor
    input7 = np.random.randn(3, 4).astype(np.float64)
    min_val7 = -1.0
    max_val7 = 1.0
    input_dict7 = {"input": input7, "min": min_val7, "max": max_val7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = clip_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('clip', generated_inputs)
