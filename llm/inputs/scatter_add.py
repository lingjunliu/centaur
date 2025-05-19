
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def scatter_add_inputs():
    list_of_inputs = []

    # Test case 1: Basic case with float input
    input1 = np.zeros((5,), dtype=np.float32)
    index1 = np.array([0, 1, 2, 0, 3], dtype=np.int64)
    src1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    dim1 = 0
    input_dict1 = {"input": input1, "dim": dim1, "index": index1, "src": src1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Test case 2: Integer input
    input2 = np.zeros((5,), dtype=np.int64)
    index2 = np.array([0, 1, 2, 0, 3], dtype=np.int64)
    src2 = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    dim2 = 0
    input_dict2 = {"input": input2, "dim": dim2, "index": index2, "src": src2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Test case 3: Multi-dimensional input
    input3 = np.zeros((2, 5), dtype=np.float32)
    index3 = np.array([[0, 1, 2, 0, 1], [1, 0, 1, 2, 0]], dtype=np.int64)
    src3 = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]], dtype=np.float32)
    dim3 = 0
    input_dict3 = {"input": input3, "dim": dim3, "index": index3, "src": src3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Test case 4: Negative values in src
    input4 = np.zeros((5,), dtype=np.float32)
    index4 = np.array([0, 1, 2, 0, 3], dtype=np.int64)
    src4 = np.array([-1.0, 2.0, -3.0, 4.0, -5.0], dtype=np.float32)
    dim4 = 0
    input_dict4 = {"input": input4, "dim": dim4, "index": index4, "src": src4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Test case 5: Different dim, reduced index range
    input5 = np.zeros((2, 4), dtype=np.float32)
    index5 = np.array([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.int64)
    src5 = np.array([[1.0, 2.0, 3.0, 4.0], [6.0, 7.0, 8.0, 9.0]], dtype=np.float32)
    dim5 = 1
    input_dict5 = {"input": input5, "dim": dim5, "index": index5, "src": src5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = scatter_add_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('scatter_add', generated_inputs)
