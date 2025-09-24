
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def is_floating_point_inputs():
    list_of_inputs = []

    # Input 1: Float tensor
    input1 = np.random.randn(3, 4).astype(np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Double tensor
    input2 = np.random.randn(2, 2, 2).astype(np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Half tensor
    input3 = np.random.randn(5).astype(np.float16)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Int tensor (should return False)
    input4 = np.random.randint(0, 10, size=(1, 5)).astype(np.int64)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Complex Float Tensor
    input5 = (np.random.randn(2, 3) + 1j * np.random.randn(2, 3)).astype(np.complex64)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Scalar Float
    input6 = np.float32(3.14)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Zero-dimensional float
    input7 = np.array(5.0).astype(np.float32)
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = is_floating_point_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('is_floating_point', generated_inputs)
