
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def fmin_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensors
    input1 = torch.randn(3, 4).numpy()
    other1 = torch.randn(3, 4).numpy()
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: Integer tensors
    input2 = torch.randint(0, 10, (2, 5)).numpy()
    other2 = torch.randint(0, 10, (2, 5)).numpy()
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: Different shapes (but still compatible)
    input3 = torch.randn(2, 3, 4).numpy()
    other3 = torch.randn(2, 3, 4).numpy()
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: One dimensional tensors
    input4 = torch.randn(10).numpy()
    other4 = torch.randn(10).numpy()
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Case 5: Negative values
    input5 = torch.randn(5, 5).numpy() - 2
    other5 = torch.randn(5, 5).numpy() - 1
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Case 6: Scalar input
    input6 = torch.randn(1).numpy()
    other6 = torch.randn(1).numpy()
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Case 7: Broadcasting
    input7 = torch.randn(5, 1).numpy()
    other7 = torch.randn(5, 5).numpy()
    input_dict7 = {"input": input7, "other": other7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Case 8: Different dtypes (float64)
    input8 = torch.randn(3, 4, dtype=torch.float64).numpy()
    other8 = torch.randn(3, 4, dtype=torch.float64).numpy()
    input_dict8 = {"input": input8, "other": other8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    return list_of_inputs

generated_inputs = fmin_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('fmin', generated_inputs)
