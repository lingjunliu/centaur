
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def mul_inputs():
    list_of_inputs = []

    # Example 1: Float tensors
    input1 = torch.randn(3, 4).numpy()
    other1 = torch.randn(3, 4).numpy()
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Example 2: Integer tensors
    input2 = torch.randint(0, 10, (2, 2)).numpy()
    other2 = torch.randint(0, 5, (2, 2)).numpy()
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Example 3: Broadcasting with a scalar
    input3 = torch.randn(5).numpy()
    other3 = 2.5
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Example 4: Broadcasting with different shapes
    input4 = torch.randn(4, 1).numpy()
    other4 = torch.randn(1, 4).numpy()
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Example 5: Negative values
    input5 = torch.randn(2, 3) * -1.0
    other5 = torch.randn(2, 3) * -0.5
    input_dict5 = {"input": input5.numpy(), "other": other5.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Example 6: Complex tensors
    input6 = (torch.randn(2, 2) + 1j * torch.randn(2, 2)).numpy()
    other6 = (torch.randn(2, 2) + 1j * torch.randn(2, 2)).numpy()
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Example 7: Different dimensions
    input7 = torch.randn(1,).numpy()
    other7 = torch.randn(1,).numpy()
    input_dict7 = {"input": input7, "other": other7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Example 8: Large tensors
    input8 = torch.randn(100, 100).numpy()
    other8 = torch.randn(100, 100).numpy()
    input_dict8 = {"input": input8, "other": other8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    return list_of_inputs

generated_inputs = mul_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('mul', generated_inputs)
