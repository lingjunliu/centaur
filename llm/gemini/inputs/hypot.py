
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def hypot_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensors
    input1 = np.array([3.0, 4.0], dtype=np.float32)
    input2 = np.array([5.0, 12.0], dtype=np.float32)
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Negative values
    input1 = np.array([-3.0, 4.0], dtype=np.float32)
    input2 = np.array([5.0, -12.0], dtype=np.float32)
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Multi-dimensional tensors
    input1 = np.array([[3.0, 4.0], [1.0, 2.0]], dtype=np.float32)
    input2 = np.array([[5.0, 12.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Different shapes (but broadcastable)
    input1 = np.array([3.0, 4.0], dtype=np.float32)
    input2 = np.array([5.0], dtype=np.float32)
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Zero values
    input1 = np.array([0.0, 4.0], dtype=np.float32)
    input2 = np.array([5.0, 0.0], dtype=np.float32)
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 6: Larger dimensions
    input1 = np.random.rand(2, 3, 4).astype(np.float32)
    input2 = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = hypot_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('hypot', generated_inputs)
