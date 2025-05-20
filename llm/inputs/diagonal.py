
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def diagonal_inputs():
    list_of_inputs = []

    # Example 1: Basic 2D tensor
    input1 = torch.randn(3, 3).numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Example 2: 2D tensor with offset
    input2 = torch.randn(4, 4).numpy()
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Example 3: 2D tensor with offset > 0
    input3 = torch.randn(5, 5).numpy()
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Example 4: 2D tensor with offset < 0
    input4 = torch.randn(6, 6).numpy()
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Example 5: 3D tensor with specified dims
    input5 = torch.randn(2, 5, 4).numpy()
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Example 6: 4D tensor
    input6 = torch.randn(2, 3, 4, 5).numpy()
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Example 7: Rectangular matrix
    input7 = torch.randn(2, 5).numpy()
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Example 8: Integer tensor
    input8 = torch.randint(0, 10, (3, 3)).numpy()
    input_dict8 = {"input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Example 9: Float64 tensor
    input9 = torch.randn(3, 3, dtype=torch.float64).numpy()
    input_dict9 = {"input": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Example 10: Complex tensor
    input10 = torch.randn(3, 3, dtype=torch.complex64).numpy()
    input_dict10 = {"input": input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = diagonal_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('diagonal', generated_inputs)
