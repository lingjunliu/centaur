
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def argmax_inputs():
    list_of_inputs = []

    # Case 1: 1D float tensor
    input_1 = torch.randn(5).numpy()
    input_dict_1 = {"input": input_1, "dim": None, "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: 2D int tensor with dim=0
    input_2 = torch.randint(-10, 10, (3, 4)).numpy()
    input_dict_2 = {"input": input_2, "dim": 0, "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: 2D float tensor with dim=1 and keepdim=True
    input_3 = torch.randn(2, 5).numpy()
    input_dict_3 = {"input": input_3, "dim": 1, "keepdim": True}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: 3D float tensor with dim=2
    input_4 = torch.randn(2, 3, 4).numpy()
    input_dict_4 = {"input": input_4, "dim": 2, "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: 4D int tensor with dim=3 and keepdim=True
    input_5 = torch.randint(-5, 5, (1, 2, 3, 4)).numpy()
    input_dict_5 = {"input": input_5, "dim": 3, "keepdim": True}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: 1D tensor with only negative values
    input_6 = torch.randn(5) * -1
    input_6 = input_6.numpy()
    input_dict_6 = {"input": input_6, "dim": None, "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Case 8: 2D tensor with all same values
    input_8 = torch.full((3,3), 5).float().numpy()
    input_dict_8 = {"input": input_8, "dim": 1, "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    return list_of_inputs

generated_inputs = argmax_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('argmax', generated_inputs)
