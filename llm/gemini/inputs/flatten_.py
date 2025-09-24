
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def flatten_inputs():
    list_of_inputs = []

    # Case 1: Basic 2D float tensor, default start and end dim
    input_tensor = np.random.randn(2, 3).astype(np.float32)
    input_dict = {"input": input_tensor, "start_dim": 0, "end_dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 3D int tensor, specified start and end dim
    input_tensor = np.random.randint(0, 10, size=(2, 3, 4)).astype(np.int32)
    input_dict = {"input": input_tensor, "start_dim": 1, "end_dim": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 4D complex tensor, negative start and end dim
    input_tensor = (np.random.randn(2, 3, 2, 2) + 1j * np.random.randn(2, 3, 2, 2)).astype(np.complex64)
    input_dict = {"input": input_tensor, "start_dim": -2, "end_dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 1D tensor, start and end dim are the same
    input_tensor = np.array([1, 2, 3, 4, 5]).astype(np.int64)
    input_dict = {"input": input_tensor, "start_dim": 0, "end_dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: 5D tensor, covering most dimensions
    input_tensor = np.random.randn(1, 2, 3, 4, 5).astype(np.float64)
    input_dict = {"input": input_tensor, "start_dim": 2, "end_dim": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: 3D tensor, negative indexing
    input_tensor = np.random.randn(2, 4, 6).astype(np.float32)
    input_dict = {"input": input_tensor, "start_dim": 0, "end_dim": -2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 7: bool tensor
    input_tensor = np.array([[True, False], [False, True]]).astype(bool)
    input_dict = {"input": input_tensor, "start_dim": 0, "end_dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = flatten_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('flatten_', generated_inputs)
