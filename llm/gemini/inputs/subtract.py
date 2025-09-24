
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def subtract_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensors
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    input2 = np.array([[0.5, 1.0], [1.5, 2.0]])
    alpha1 = 1.0
    input_dict1 = {"input": input1, "other": input2, "alpha": alpha1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: Integer tensors with alpha=1
    input3 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input4 = np.array([[0, 1], [1, 2]], dtype=np.int32)
    alpha2 = 1
    input_dict2 = {"input": input3, "other": input4, "alpha": alpha2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: Negative values and different shapes (broadcasting)
    input5 = np.array([[-1.0, -2.0], [-3.0, -4.0]])
    input6 = np.array([1.0, 2.0])
    alpha3 = 1.0
    input_dict3 = {"input": input5, "other": input6, "alpha": alpha3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: Higher dimensions (float)
    input7 = np.random.rand(2, 3, 4)
    input8 = np.random.rand(2, 3, 4)
    alpha4 = 0.5
    input_dict4 = {"input": input7, "other": input8, "alpha": alpha4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Case 5: Scalar other (float)
    input9 = np.array([[5.0, 6.0], [7.0, 8.0]])
    input10 = np.array(2.0)
    alpha5 = 1.0
    input_dict5 = {"input": input9, "other": input10, "alpha": alpha5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = subtract_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('subtract', generated_inputs)
