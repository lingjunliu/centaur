
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def broadcast_to_inputs():
    list_of_inputs = []

    # Case 1: Simple 1D broadcast
    input_tensor = np.array([1, 2, 3])
    shape = [3, 3]
    input_dict = {"input": input_tensor, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Broadcast a scalar to a multi-dimensional tensor
    input_tensor = np.array(5)
    shape = [2, 3, 4]
    input_dict = {"input": input_tensor, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Broadcast a 2D tensor to a 3D tensor
    input_tensor = np.array([[1, 2], [3, 4]])
    shape = [2, 2, 2]
    input_dict = {"input": input_tensor, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Broadcast a tensor with compatible dimensions
    input_tensor = np.array([[1, 2, 3]])
    shape = [2, 1, 3]
    input_dict = {"input": input_tensor, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Broadcast a tensor with different data type (int)
    input_tensor = np.array([1, 2, 3], dtype=np.int32)
    shape = [3, 3]
    input_dict = {"input": input_tensor, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Broadcast a tensor with different data type (float)
    input_tensor = np.array([1.0, 2.0], dtype=np.float64)
    shape = [3, 2]
    input_dict = {"input": input_tensor, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Broadcast a 3D tensor to a 4D tensor
    input_tensor = np.random.rand(2, 3, 4)
    shape = [5, 2, 3, 4]
    input_dict = {"input": input_tensor, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: Broadcast along multiple dimensions
    input_tensor = np.array([[[1], [2]]])
    shape = [2, 1, 2, 3]
    input_dict = {"input": input_tensor, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = broadcast_to_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('broadcast_to', generated_inputs)
