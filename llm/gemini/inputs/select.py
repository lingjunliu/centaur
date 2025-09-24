
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def torch_select_inputs():
    list_of_inputs = []

    # Input 1: 2D float tensor, dim=0, index=0
    input1 = torch.randn(3, 4).numpy()
    input_dict1 = {"input": input1, "dim": 0, "index": 0}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 3D int tensor, dim=1, index=1
    input2 = torch.randint(0, 10, (2, 5, 3)).numpy()
    input_dict2 = {"input": input2, "dim": 1, "index": 1}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 4D complex tensor, dim=2, index=-1
    input3 = torch.randn(2, 3, 4, 5, dtype=torch.complex64).numpy()
    input_dict3 = {"input": input3, "dim": 2, "index": -1}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 1D tensor, dim=0, index=0
    input4 = torch.arange(5).numpy()
    input_dict4 = {"input": input4, "dim": 0, "index": 0}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 3D float tensor with negative index
    input5 = torch.randn(4, 6, 8).numpy()
    input_dict5 = {"input": input5, "dim": 1, "index": -2}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: 2D bool tensor
    input6 = torch.randint(0, 2, (3, 4)).bool().numpy()
    input_dict6 = {"input": input6, "dim": 0, "index": 1}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: 5D float tensor
    input7 = torch.randn(2, 3, 4, 5, 6).numpy()
    input_dict7 = {"input": input7, "dim": 3, "index": 2}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs = torch_select_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('select', generated_inputs)
