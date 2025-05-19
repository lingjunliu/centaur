
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def bmm_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float tensors
    input1 = np.random.randn(10, 3, 4).astype(np.float32)
    mat2_1 = np.random.randn(10, 4, 5).astype(np.float32)
    input_dict1 = {"input": input1, "mat2": mat2_1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Different batch size and dimensions
    input2 = np.random.randn(5, 2, 3).astype(np.float64)
    mat2_2 = np.random.randn(5, 3, 6).astype(np.float64)
    input_dict2 = {"input": input2, "mat2": mat2_2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Using negative values
    input3 = np.random.randn(2, 5, 5).astype(np.float32) * -1
    mat2_3 = np.random.randn(2, 5, 5).astype(np.float32) * -1
    input_dict3 = {"input": input3, "mat2": mat2_3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: Different data type (float16)
    input4 = np.random.randn(3, 4, 2).astype(np.float16)
    mat2_4 = np.random.randn(3, 2, 3).astype(np.float16)
    input_dict4 = {"input": input4, "mat2": mat2_4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Larger matrices
    input5 = np.random.randn(1, 128, 256).astype(np.float32)
    mat2_5 = np.random.randn(1, 256, 512).astype(np.float32)
    input_dict5 = {"input": input5, "mat2": mat2_5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Different batch size
    input6 = np.random.randn(32, 8, 16).astype(np.float32)
    mat2_6 = np.random.randn(32, 16, 32).astype(np.float32)
    input_dict6 = {"input": input6, "mat2": mat2_6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Using doubles
    input7 = np.random.randn(4, 10, 10).astype(np.float64)
    mat2_7 = np.random.randn(4, 10, 10).astype(np.float64)
    input_dict7 = {"input": input7, "mat2": mat2_7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = bmm_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('bmm', generated_inputs)
