
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def bincount_inputs():
    list_of_inputs = []

    # Example 1: Basic integer input
    input1 = np.array([1, 2, 2, 3, 3, 3], dtype=np.int64)
    weights1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
    minlength1 = 0
    input_dict1 = {"input": input1, "weights": weights1, "minlength": minlength1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Example 2: Input with minlength
    input2 = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    weights2 = np.array([1, 1, 1, 1, 1], dtype=np.float64)
    minlength2 = 7
    input_dict2 = {"input": input2, "weights": weights2, "minlength": minlength2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Example 3: Input with zero weights
    input3 = np.array([0, 1, 0, 1, 2], dtype=np.int64)
    weights3 = np.array([0, 0, 0, 0, 0], dtype=np.float32)
    minlength3 = 0
    input_dict3 = {"input": input3, "weights": weights3, "minlength": minlength3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Example 4: Input with different weights
    input4 = np.array([0, 1, 2, 0, 1, 2, 0], dtype=np.int32)
    weights4 = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=np.float64)
    minlength4 = 0
    input_dict4 = {"input": input4, "weights": weights4, "minlength": minlength4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Example 5: Input with one element
    input5 = np.array([5], dtype=np.int64)
    weights5 = np.array([2.0], dtype=np.float32)
    minlength5 = 10
    input_dict5 = {"input": input5, "weights": weights5, "minlength": minlength5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Example 6: Input with no weights
    input6 = np.array([0, 1, 2, 1, 0], dtype=np.int32)
    weights6 = None
    minlength6 = 0
    input_dict6 = {"input": input6, "weights": weights6, "minlength": minlength6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Example 7: Input with smaller weights
    input7 = np.array([2, 1, 2, 0], dtype=np.int64)
    weights7 = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    minlength7 = 5
    input_dict7 = {"input": input7, "weights": weights7, "minlength": minlength7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs = bincount_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('bincount', generated_inputs)
