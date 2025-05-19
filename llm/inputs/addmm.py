
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def addmm_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensors
    input1 = np.random.randn(3, 5).astype(np.float32)
    mat1_1 = np.random.randn(3, 4).astype(np.float32)
    mat2_1 = np.random.randn(4, 5).astype(np.float32)
    input_dict1 = {"input": input1, "mat1": mat1_1, "mat2": mat2_1, "beta": 1.0, "alpha": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Int tensors with different beta and alpha
    input2 = np.random.randint(-5, 5, size=(2, 3)).astype(np.int32)
    mat1_2 = np.random.randint(-5, 5, size=(2, 4)).astype(np.int32)
    mat2_2 = np.random.randint(-5, 5, size=(4, 3)).astype(np.int32)
    input_dict2 = {"input": input2, "mat1": mat1_2, "mat2": mat2_2, "beta": 0.5, "alpha": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Negative values and zero beta
    input3 = np.random.randn(4, 2).astype(np.float64) * -1
    mat1_3 = np.random.randn(4, 3).astype(np.float64) * -1
    mat2_3 = np.random.randn(3, 2).astype(np.float64) * -1
    input_dict3 = {"input": input3, "mat1": mat1_3, "mat2": mat2_3, "beta": 0.0, "alpha": 1.5}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: Larger matrices
    input4 = np.random.randn(10, 10).astype(np.float32)
    mat1_4 = np.random.randn(10, 5).astype(np.float32)
    mat2_4 = np.random.randn(5, 10).astype(np.float32)
    input_dict4 = {"input": input4, "mat1": mat1_4, "mat2": mat2_4, "beta": 0.8, "alpha": 0.7}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Complex tensors
    input5 = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)).astype(np.complex64)
    mat1_5 = (np.random.randn(2, 3) + 1j * np.random.randn(2, 3)).astype(np.complex64)
    mat2_5 = (np.random.randn(3, 2) + 1j * np.random.randn(3, 2)).astype(np.complex64)
    input_dict5 = {"input": input5, "mat1": mat1_5, "mat2": mat2_5, "beta": 1.0, "alpha": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Different shapes
    input6 = np.random.randn(5, 7).astype(np.float32)
    mat1_6 = np.random.randn(5, 2).astype(np.float32)
    mat2_6 = np.random.randn(2, 7).astype(np.float32)
    input_dict6 = {"input": input6, "mat1": mat1_6, "mat2": mat2_6, "beta": 0.2, "alpha": 0.9}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs = addmm_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('addmm', generated_inputs)
