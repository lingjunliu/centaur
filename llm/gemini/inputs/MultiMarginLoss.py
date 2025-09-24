
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def multimarginloss_inputs():
    list_of_inputs = []

    # Example 1: Basic case
    input_1 = np.array([[1.0, -0.5, 0.2], [2.0, 0.3, -0.1]], dtype=np.float32)
    target_1 = np.array([0, 2], dtype=np.int64)
    input_dict_1 = {"input": input_1, "target": target_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Example 2: Different input size and target values
    input_2 = np.array([[0.5, 1.2, -0.8, 0.1], [-1.0, 0.4, 2.0, -0.5]], dtype=np.float32)
    target_2 = np.array([3, 1], dtype=np.int64)
    input_dict_2 = {"input": input_2, "target": target_2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Example 3: More examples
    input_3 = np.array([[1.5, 0.2, -0.7, 0.9, -0.3], [0.1, -1.2, 0.5, -0.6, 1.8]], dtype=np.float32)
    target_3 = np.array([4, 0], dtype=np.int64)
    input_dict_3 = {"input": input_3, "target": target_3}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Example 4: Larger input and target
    input_4 = np.array([[0.8, -0.2, 1.1, -0.5, 0.3, -1.0, 0.6], [-0.4, 1.3, -0.9, 0.2, -0.7, 0.1, 1.2]], dtype=np.float32)
    target_4 = np.array([6, 1], dtype=np.int64)
    input_dict_4 = {"input": input_4, "target": target_4}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Example 5: Different batch size
    input_5 = np.array([[0.3, -0.1, 0.5], [-0.2, 0.4, 0.1], [0.6, 0.2, -0.3]], dtype=np.float32)
    target_5 = np.array([1, 2, 0], dtype=np.int64)
    input_dict_5 = {"input": input_5, "target": target_5}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Example 6: all values positive
    input_6 = np.array([[0.3, 0.1, 0.5], [0.2, 0.4, 0.1], [0.6, 0.2, 0.3]], dtype=np.float32)
    target_6 = np.array([1, 2, 0], dtype=np.int64)
    input_dict_6 = {"input": input_6, "target": target_6}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Example 7: Using float64
    input_7 = np.array([[1.0, -0.5, 0.2], [2.0, 0.3, -0.1]], dtype=np.float64)
    target_7 = np.array([0, 2], dtype=np.int64)
    input_dict_7 = {"input": input_7, "target": target_7}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    return list_of_inputs

generated_inputs = multimarginloss_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('MultiMarginLoss', generated_inputs)
