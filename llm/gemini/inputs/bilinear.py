
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def bilinear_inputs():
    list_of_inputs = []

    # Input 1
    input1 = np.random.randn(2, 3).astype(np.float32)
    input2 = np.random.randn(2, 4).astype(np.float32)
    weight = np.random.randn(5, 3, 4).astype(np.float32)
    bias = np.random.randn(5).astype(np.float32)
    input_dict = {"input1": input1, "input2": input2, "weight": weight, "bias": bias}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input1 = np.random.randn(1, 5).astype(np.float64)
    input2 = np.random.randn(1, 2).astype(np.float64)
    weight = np.random.randn(3, 5, 2).astype(np.float64)
    bias = np.random.randn(3).astype(np.float64)
    input_dict = {"input1": input1, "input2": input2, "weight": weight, "bias": bias}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3 - negative values
    input1 = np.random.randn(3, 2).astype(np.float32) * -1
    input2 = np.random.randn(3, 6).astype(np.float32) * -1
    weight = np.random.randn(4, 2, 6).astype(np.float32)
    bias = np.random.randn(4).astype(np.float32) * -1
    input_dict = {"input1": input1, "input2": input2, "weight": weight, "bias": bias}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4 - different sizes
    input1 = np.random.randn(4, 1).astype(np.float32)
    input2 = np.random.randn(4, 7).astype(np.float32)
    weight = np.random.randn(2, 1, 7).astype(np.float32)
    bias = np.random.randn(2).astype(np.float32)
    input_dict = {"input1": input1, "input2": input2, "weight": weight, "bias": bias}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 - Larger sizes
    input1 = np.random.randn(5, 10).astype(np.float32)
    input2 = np.random.randn(5, 8).astype(np.float32)
    weight = np.random.randn(7, 10, 8).astype(np.float32)
    bias = np.random.randn(7).astype(np.float32)
    input_dict = {"input1": input1, "input2": input2, "weight": weight, "bias": bias}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = bilinear_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('bilinear', generated_inputs)
