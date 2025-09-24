
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def concatenate_inputs():
    list_of_inputs = []

    # Test case 1: Basic 2D tensors, axis=0
    tensors1 = [np.random.rand(2, 3).astype(np.float32), np.random.rand(3, 3).astype(np.float32)]
    input_dict1 = {"tensors": tensors1, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Test case 2: Basic 2D tensors, axis=1
    tensors2 = [np.random.rand(2, 3).astype(np.float32), np.random.rand(2, 4).astype(np.float32)]
    input_dict2 = {"tensors": tensors2, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Test case 3: 3D tensors, axis=0
    tensors3 = [np.random.rand(2, 3, 4).astype(np.float32), np.random.rand(1, 3, 4).astype(np.float32)]
    input_dict3 = {"tensors": tensors3, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Test case 4: 3D tensors, axis=1
    tensors4 = [np.random.rand(2, 3, 4).astype(np.float32), np.random.rand(2, 2, 4).astype(np.float32)]
    input_dict4 = {"tensors": tensors4, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Test case 5: 3D tensors, axis=2
    tensors5 = [np.random.rand(2, 3, 4).astype(np.float32), np.random.rand(2, 3, 5).astype(np.float32)]
    input_dict5 = {"tensors": tensors5, "axis": 2}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Test case 6: Different data types (int32)
    tensors6 = [np.random.randint(0, 10, size=(2, 3), dtype=np.int32), np.random.randint(0, 10, size=(3, 3), dtype=np.int32)]
    input_dict6 = {"tensors": tensors6, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Test case 7: Different data types (int64)
    tensors7 = [np.random.randint(0, 10, size=(2, 3), dtype=np.int64), np.random.randint(0, 10, size=(3, 3), dtype=np.int64)]
    input_dict7 = {"tensors": tensors7, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Test case 8: Different shapes and dtypes (float64 and int32)
    tensors8 = [np.random.rand(2, 3).astype(np.float64), np.random.randint(0, 10, size=(2, 3), dtype=np.int32).astype(np.float64)]
    input_dict8 = {"tensors": tensors8, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Test case 9:  1D tensors
    tensors9 = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    input_dict9 = {"tensors": tensors9, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    return list_of_inputs

generated_inputs = concatenate_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('concatenate', generated_inputs)
