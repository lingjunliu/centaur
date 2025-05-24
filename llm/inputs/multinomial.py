
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def multinomial_inputs():
    generated_inputs = []

    # Example 1: Basic vector input
    input1 = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    num_samples1 = 2
    input_dict1 = {"input": input1, "num_samples": num_samples1}
    generated_inputs.append(copy.deepcopy(input_dict1))

    # Example 2: Matrix input
    input2 = np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]], dtype=np.float32)
    num_samples2 = 2
    input_dict2 = {"input": input2, "num_samples": num_samples2}
    generated_inputs.append(copy.deepcopy(input_dict2))

    # Example 3: Different number of samples
    input3 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    num_samples3 = 3
    input_dict3 = {"input": input3, "num_samples": num_samples3}
    generated_inputs.append(copy.deepcopy(input_dict3))

    # Example 4: Larger input vector
    input4 = np.array([0.05, 0.15, 0.2, 0.1, 0.05, 0.25, 0.05, 0.1], dtype=np.float64)
    num_samples4 = 4
    input_dict4 = {"input": input4, "num_samples": num_samples4}
    generated_inputs.append(copy.deepcopy(input_dict4))

    # Example 5: Input with a zero probability
    input5 = np.array([0.0, 0.5, 0.5], dtype=np.float32)
    num_samples5 = 1
    input_dict5 = {"input": input5, "num_samples": num_samples5}
    generated_inputs.append(copy.deepcopy(input_dict5))

    return generated_inputs

generated_inputs = multinomial_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('multinomial', generated_inputs)
