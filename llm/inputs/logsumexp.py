
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def logsumexp_inputs():
    list_of_inputs = []

    # Input 1: 2D tensor, dim=1, keepdim=False
    input1 = torch.randn(3, 4).numpy()
    input_dict1 = {"input": input1, "dim": 1, "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor, dim=0, keepdim=True
    input2 = torch.randn(5, 2).numpy()
    input_dict2 = {"input": input2, "dim": 0, "keepdim": True}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D tensor, dim=2, keepdim=False
    input3 = torch.randn(2, 3, 4).numpy()
    input_dict3 = {"input": input3, "dim": 2, "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 3D tensor, dim=(1,2), keepdim=True
    input4 = torch.randn(2, 3, 4).numpy()
    input_dict4 = {"input": input4, "dim": (1, 2), "keepdim": True}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 1D tensor, dim=0, keepdim=False (valid as dim is ignored for 1D)
    input5 = torch.randn(5).numpy()
    input_dict5 = {"input": input5, "dim": 0, "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 2D tensor with negative values, dim=1, keepdim=False
    input6 = torch.randn(3, 4).numpy() * -1
    input_dict6 = {"input": input6, "dim": 1, "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: 4D tensor, dim=3, keepdim=True
    input7 = torch.randn(2, 3, 4, 5).numpy()
    input_dict7 = {"input": input7, "dim": 3, "keepdim": True}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = logsumexp_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('logsumexp', generated_inputs)
