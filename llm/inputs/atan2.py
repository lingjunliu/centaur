
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def atan2_inputs():
    list_of_inputs = []

    # Test case 1: Basic float tensors
    input1 = np.random.randn(3, 4).astype(np.float32)
    other1 = np.random.randn(3, 4).astype(np.float32)
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Test case 2: Integer tensors
    input2 = np.random.randint(-5, 5, size=(2, 2)).astype(np.int32)
    other2 = np.random.randint(-5, 5, size=(2, 2)).astype(np.int32)
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Test case 3: Negative values
    input3 = np.random.randn(5).astype(np.float64) * -1
    other3 = np.random.randn(5).astype(np.float64) * -1
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Test case 4: Different shapes (but broadcastable)
    input4 = np.random.randn(2, 3, 4).astype(np.float32)
    other4 = np.random.randn(4).astype(np.float32)
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Test case 5: Scalar values
    input5 = np.random.randn(1).astype(np.float32)[0]
    other5 = np.random.randn(1).astype(np.float32)[0]
    input_dict5 = {"input": np.array(input5), "other": np.array(other5)}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Test case 6: One dimensional arrays
    input6 = np.random.randn(10).astype(np.float32)
    other6 = np.random.randn(10).astype(np.float32)
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Test case 7: Zero values
    input7 = np.zeros((3, 3)).astype(np.float32)
    other7 = np.ones((3, 3)).astype(np.float32)
    input_dict7 = {"input": input7, "other": other7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs = atan2_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('atan2', generated_inputs)
