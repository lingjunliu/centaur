
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def mm_inputs():
    generated_inputs = []

    # Input 1: Basic float tensors
    input1 = np.random.rand(3, 4).astype(np.float32)
    mat2_1 = np.random.rand(4, 5).astype(np.float32)
    generated_inputs.append({"input": input1, "mat2": mat2_1})

    # Input 2: Float tensors with different shapes
    input2 = np.random.rand(1, 5).astype(np.float32)
    mat2_2 = np.random.rand(5, 1).astype(np.float32)
    generated_inputs.append({"input": input2, "mat2": mat2_2})

    # Input 3: Float tensors with negative values
    input3 = np.random.randn(2, 3).astype(np.float32)
    mat2_3 = np.random.randn(3, 2).astype(np.float32)
    generated_inputs.append({"input": input3, "mat2": mat2_3})

    # Input 4: Double tensors
    input4 = np.random.rand(3, 4).astype(np.float64)
    mat2_4 = np.random.rand(4, 5).astype(np.float64)
    generated_inputs.append({"input": input4, "mat2": mat2_4})

    # Input 5: Complex tensors
    input5 = (np.random.rand(2, 3) + 1j * np.random.rand(2, 3)).astype(np.complex64)
    mat2_5 = (np.random.rand(3, 2) + 1j * np.random.rand(3, 2)).astype(np.complex64)
    generated_inputs.append({"input": input5, "mat2": mat2_5})

    # Input 6: Larger tensors
    input6 = np.random.rand(10, 20).astype(np.float32)
    mat2_6 = np.random.rand(20, 10).astype(np.float32)
    generated_inputs.append({"input": input6, "mat2": mat2_6})

    # Input 7: Single element tensors
    input7 = np.random.rand(1, 1).astype(np.float32)
    mat2_7 = np.random.rand(1, 1).astype(np.float32)
    generated_inputs.append({"input": input7, "mat2": mat2_7})

    # Input 8: Tensors with zero values
    input8 = np.zeros((3, 4), dtype=np.float32)
    mat2_8 = np.zeros((4, 5), dtype=np.float32)
    generated_inputs.append({"input": input8, "mat2": mat2_8})

    return generated_inputs

generated_inputs = mm_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('mm', generated_inputs)
