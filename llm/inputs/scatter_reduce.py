
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def scatter_reduce_inputs():
    list_of_inputs = []

    # Test case 1: Basic sum reduction
    input1 = np.zeros((5,), dtype=np.float32)
    index1 = np.array([0, 1, 2, 0, 3], dtype=np.int64)
    src1 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    dim1 = 0
    reduce1 = "sum"
    include_self1 = False
    input_dict1 = {"input": input1, "dim": dim1, "index": index1, "src": src1, "reduce": reduce1, "include_self": include_self1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Test case 2:  Multi-dimensional input with mean reduction
    input2 = np.zeros((2, 3), dtype=np.float32)
    index2 = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int64)
    src2 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    dim2 = 1
    reduce2 = "mean"
    include_self2 = False
    input_dict2 = {"input": input2, "dim": dim2, "index": index2, "src": src2, "reduce": reduce2, "include_self": include_self2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Test case 3: Use include_self = True
    input3 = np.ones((5,), dtype=np.float32)
    index3 = np.array([0, 1, 2, 0, 3], dtype=np.int64)
    src3 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    dim3 = 0
    reduce3 = "sum"
    include_self3 = True
    input_dict3 = {"input": input3, "dim": dim3, "index": index3, "src": src3, "reduce": reduce3, "include_self": include_self3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Test case 4: min reduction
    input4 = np.array([5, 5, 5, 5, 5], dtype=np.float32)
    index4 = np.array([0, 1, 2, 0, 3], dtype=np.int64)
    src4 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    dim4 = 0
    reduce4 = "min"
    include_self4 = True
    input_dict4 = {"input": input4, "dim": dim4, "index": index4, "src": src4, "reduce": reduce4, "include_self": include_self4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Test case 5: max reduction
    input5 = np.array([0, 0, 0, 0, 0], dtype=np.float32)
    index5 = np.array([0, 1, 2, 0, 3], dtype=np.int64)
    src5 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    dim5 = 0
    reduce5 = "max"
    include_self5 = True
    input_dict5 = {"input": input5, "dim": dim5, "index": index5, "src": src5, "reduce": reduce5, "include_self": include_self5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Test case 6: prod reduction
    input6 = np.ones((5,), dtype=np.float32)
    index6 = np.array([0, 1, 2, 0, 3], dtype=np.int64)
    src6 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    dim6 = 0
    reduce6 = "prod"
    include_self6 = True
    input_dict6 = {"input": input6, "dim": dim6, "index": index6, "src": src6, "reduce": reduce6, "include_self": include_self6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Test case 7: float64
    input7 = np.zeros((5,), dtype=np.float64)
    index7 = np.array([0, 1, 2, 0, 3], dtype=np.int64)
    src7 = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    dim7 = 0
    reduce7 = "sum"
    include_self7 = False
    input_dict7 = {"input": input7, "dim": dim7, "index": index7, "src": src7, "reduce": reduce7, "include_self": include_self7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = scatter_reduce_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('scatter_reduce', generated_inputs)
