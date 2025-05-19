
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def mean_inputs():
    list_of_inputs = []

    # Case 1: 1D float tensor, no dim specified
    input1 = torch.randn(5).numpy()
    input_dict1 = {"input": input1, "dim": None, "keepdim": False, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: 2D int tensor, dim=0, keepdim=True
    input2 = torch.randint(-10, 10, (3, 4)).numpy()
    input_dict2 = {"input": input2, "dim": 0, "keepdim": True, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: 3D float tensor, dim=2, keepdim=False
    input3 = torch.randn(2, 3, 4).numpy()
    input_dict3 = {"input": input3, "dim": 2, "keepdim": False, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: 2D complex tensor, dim=1, keepdim=True
    input4 = (torch.randn(2, 2) + 1j * torch.randn(2, 2)).numpy()
    input_dict4 = {"input": input4, "dim": 1, "keepdim": True, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Case 5: 0D float tensor (scalar), no dim
    input5 = torch.randn(1).numpy().item()
    input_dict5 = {"input": np.array(input5), "dim": None, "keepdim": False, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = mean_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('mean', generated_inputs)
