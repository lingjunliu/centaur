
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def softmax_inputs():
    list_of_inputs = []

    # Case 1: 1D tensor, float32
    input_1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict_1 = {"input": input_1, "dim": 0, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: 2D tensor, float64, negative values
    input_2 = np.array([[-1.0, 0.0, 1.0], [2.0, -2.0, 0.5]], dtype=np.float64)
    input_dict_2 = {"input": input_2, "dim": 1, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: 3D tensor, float32, dim=2
    input_3 = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict_3 = {"input": input_3, "dim": 2, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: 2D tensor, float16, dim=0, specify dtype
    input_4 = np.array([[0.5, 0.2], [0.8, 0.1]], dtype=np.float16)
    input_dict_4 = {"input": input_4, "dim": 0, "dtype": torch.float16}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: 4D tensor, float64, dim=3
    input_5 = np.random.randn(1, 2, 3, 4).astype(np.float64)
    input_dict_5 = {"input": input_5, "dim": 3, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: 2D tensor, mixed positive and negative
    input_6 = np.array([[-1.5, 2.5], [0.0, -0.5]], dtype=np.float32)
    input_dict_6 = {"input": input_6, "dim": 1, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Case 7: 1D tensor with a single element
    input_7 = np.array([5.0], dtype=np.float32)
    input_dict_7 = {"input": input_7, "dim": 0, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    return list_of_inputs

generated_inputs = softmax_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('softmax', generated_inputs)
