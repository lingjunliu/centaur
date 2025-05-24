
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def cumprod_inputs():
    list_of_inputs = []

    # Example 1: 1D tensor, positive values
    input_1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    dim_1 = 0
    input_dict_1 = {"input": input_1, "dim": dim_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Example 2: 2D tensor, negative values
    input_2 = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]])
    dim_2 = 1
    input_dict_2 = {"input": input_2, "dim": dim_2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Example 3: 3D tensor, mixed values, dim=0
    input_3 = np.random.randn(2, 3, 4)
    dim_3 = 0
    input_dict_3 = {"input": input_3, "dim": dim_3}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Example 4: 3D tensor, mixed values, dim=1
    input_4 = np.random.randn(2, 3, 4)
    dim_4 = 1
    input_dict_4 = {"input": input_4, "dim": dim_4}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Example 5: 3D tensor, mixed values, dim=2
    input_5 = np.random.randn(2, 3, 4)
    dim_5 = 2
    input_dict_5 = {"input": input_5, "dim": dim_5}
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    
    # Example 6: 4D tensor, mixed values, dim=3
    input_6 = np.random.randn(2, 3, 4, 5)
    dim_6 = 3
    input_dict_6 = {"input": input_6, "dim": dim_6}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Example 7: Int tensor
    input_7 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    dim_7 = 0
    input_dict_7 = {"input": input_7, "dim": dim_7}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Example 8: tensor with zeros
    input_8 = np.array([0.0, 1.0, 2.0, 0.0, 3.0])
    dim_8 = 0
    input_dict_8 = {"input": input_8, "dim": dim_8}
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Example 9: empty tensor
    input_9 = np.array([])
    dim_9 = 0
    input_dict_9 = {"input": input_9, "dim": dim_9}
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    return list_of_inputs

generated_inputs = cumprod_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('cumprod', generated_inputs)
