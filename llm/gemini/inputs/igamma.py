
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def igamma_inputs():
    list_of_inputs = []

    # Test case 1: Basic float tensors
    input1 = np.random.rand(3, 4).astype(np.float32)
    other1 = np.random.rand(3, 4).astype(np.float32)
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Test case 2: Integer tensors (should be cast to float)
    input2 = np.random.randint(1, 10, size=(2, 2)).astype(np.float32)
    other2 = np.random.randint(1, 10, size=(2, 2)).astype(np.float32)
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Test case 3: Scalar inputs
    input3 = np.array(2.5).astype(np.float64)
    other3 = np.array(1.5).astype(np.float64)
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Test case 4: Different shapes (but compatible)
    input4 = np.random.rand(5, 1).astype(np.float32)
    other4 = np.random.rand(1, 5).astype(np.float32)
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Test case 5: Larger tensors with broadcasting
    input5 = np.random.rand(2, 3, 4).astype(np.float32)
    other5 = np.random.rand(3, 4).astype(np.float32)
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Test case 6: 0-dimensional array
    input6 = np.array(np.random.rand()).astype(np.float32)
    other6 = np.array(np.random.rand()).astype(np.float32)
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs = igamma_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('igamma', generated_inputs)
