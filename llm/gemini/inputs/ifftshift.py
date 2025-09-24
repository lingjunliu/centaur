
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def ifftshift_inputs():
    list_of_inputs = []

    # Case 1: 1D float tensor, default dim
    input1 = torch.randn(5).numpy()
    input_dict1 = {"input": input1, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: 2D int tensor, dim=0
    input2 = torch.randint(-5, 5, (4, 4)).numpy()
    input_dict2 = {"input": input2, "dim": (0,)}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: 3D complex tensor, dim=(1,2)
    input3 = torch.randn(2, 3, 4, dtype=torch.complex64).numpy()
    input_dict3 = {"input": input3, "dim": (1, 2)}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: 4D float tensor with negative values, dim=2
    input4 = torch.randn(1, 2, 3, 4).numpy()
    input_dict4 = {"input": input4, "dim": (2,)}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Case 5: 2D float tensor, dim=(0, 1)
    input5 = torch.randn(6, 6).numpy()
    input_dict5 = {"input": input5, "dim": (0, 1)}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Case 6: 1D int tensor with different size
    input6 = torch.randint(-10, 10, (10,)).numpy()
    input_dict6 = {"input": input6, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Case 7: 3D float tensor, dim=0
    input7 = torch.randn(3, 5, 7).numpy()
    input_dict7 = {"input": input7, "dim": (0,)}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Case 8: 2D complex tensor, dim=1
    input8 = torch.randn(4, 5, dtype=torch.complex128).numpy()
    input_dict8 = {"input": input8, "dim": (1,)}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    return list_of_inputs

generated_inputs = ifftshift_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('ifftshift', generated_inputs)
