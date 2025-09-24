
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def prelu_inputs():
    list_of_inputs = []

    # Input 1: Basic float input and weight
    input1 = np.random.randn(3, 4).astype(np.float32)
    weight1 = np.array([0.25], dtype=np.float32)
    input_dict1 = {"input": input1, "weight": weight1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Input with negative values and a different alpha
    input2 = np.random.randn(2, 2, 2).astype(np.float32)
    weight2 = np.array([-0.1], dtype=np.float32)
    input_dict2 = {"input": input2, "weight": weight2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3:  1D input with a single alpha value
    input3 = np.random.randn(5).astype(np.float32)
    weight3 = np.array([0.01], dtype=np.float32)
    input_dict3 = {"input": input3, "weight": weight3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4:  Larger input with a different alpha
    input4 = np.random.randn(1, 3, 28, 28).astype(np.float32)
    weight4 = np.array([0.0], dtype=np.float32)
    input_dict4 = {"input": input4, "weight": weight4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Multiple alpha values for each channel (requires input to have channels)
    input5 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    weight5 = np.array([0.25, -0.25, 0.0], dtype=np.float32)
    input_dict5 = {"input": input5, "weight": weight5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6:  Scalar input
    input6 = np.array(1.5, dtype=np.float32)
    weight6 = np.array([0.3], dtype=np.float32)
    input_dict6 = {"input": input6, "weight": weight6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    return list_of_inputs

generated_inputs = prelu_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('prelu', generated_inputs)
