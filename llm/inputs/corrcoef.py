
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def corrcoef_inputs():
    generated_inputs = []

    # Example 1: Basic 2D tensor
    x = np.array([[0, 1, 2], [2, 1, 0]])
    input_dict = {"input": x}
    generated_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Random 2D tensor with float values
    x = np.random.randn(2, 4)
    input_dict = {"input": x}
    generated_inputs.append(copy.deepcopy(input_dict))

    # Example 3: 1D tensor
    x = np.random.randn(5)
    input_dict = {"input": x}
    generated_inputs.append(copy.deepcopy(input_dict))

    # Example 4: Scalar tensor
    x = np.array(5)
    input_dict = {"input": x}
    generated_inputs.append(copy.deepcopy(input_dict))
    
    # Example 5: Tensor with negative values
    x = np.array([[-1, 2, -3], [4, -5, 6]])
    input_dict = {"input": x}
    generated_inputs.append(copy.deepcopy(input_dict))

    # Example 6: Larger 2D tensor
    x = np.random.rand(10, 20)
    input_dict = {"input": x}
    generated_inputs.append(copy.deepcopy(input_dict))
    
    # Example 7: Integer tensor
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    input_dict = {"input": x}
    generated_inputs.append(copy.deepcopy(input_dict))

    return generated_inputs

generated_inputs = corrcoef_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('corrcoef', generated_inputs)
