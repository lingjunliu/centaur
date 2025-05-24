
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def add_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensors
    input1 = torch.randn(4).numpy()
    other1 = torch.randn(4).numpy()
    input_dict1 = {"input": input1, "other": other1, "alpha": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: Integer tensors with different shapes (broadcasting)
    input2 = torch.randint(0, 10, (2, 3)).numpy()
    other2 = torch.randint(0, 10, (1, 3)).numpy()
    input_dict2 = {"input": input2, "other": other2, "alpha": 1}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: Scalar 'other' value
    input3 = torch.randn(5).numpy()
    other3 = 5.0
    input_dict3 = {"input": input3, "other": other3, "alpha": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Case 4: Negative values and alpha
    input4 = torch.randn(3, 3).numpy()
    other4 = torch.randn(3, 3).numpy()
    input_dict4 = {"input": input4, "other": other4, "alpha": -1.0}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Case 5: Complex tensors
    input5 = (torch.randn(2) + 1j * torch.randn(2)).numpy()
    other5 = (torch.randn(2) + 1j * torch.randn(2)).numpy()
    input_dict5 = {"input": input5, "other": other5, "alpha": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Case 6: Higher dimensional tensors
    input6 = torch.randn(2, 3, 4).numpy()
    other6 = torch.randn(2, 3, 4).numpy()
    input_dict6 = {"input": input6, "other": other6, "alpha": 0.75}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Case 7: Integer input with float other
    input7 = torch.randint(0, 5, (2, 2)).numpy()
    other7 = torch.randn(2, 2).numpy()
    input_dict7 = {"input": input7, "other": other7, "alpha": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = add_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('add', generated_inputs)
