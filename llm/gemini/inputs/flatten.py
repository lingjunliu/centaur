
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def flatten_inputs():
    list_of_inputs = []

    # Test case 1: Basic 2D tensor
    input1 = torch.randn(2, 3).numpy()
    input_dict1 = {"input": input1, "start_dim": 0, "end_dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Test case 2: 3D tensor, flatten from dim 1
    input2 = torch.randn(2, 3, 4).numpy()
    input_dict2 = {"input": input2, "start_dim": 1, "end_dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Test case 3: 4D tensor, flatten a middle range of dimensions
    input3 = torch.randn(2, 3, 4, 5).numpy()
    input_dict3 = {"input": input3, "start_dim": 1, "end_dim": 2}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Test case 4: 1D tensor
    input4 = torch.randn(5).numpy()
    input_dict4 = {"input": input4, "start_dim": 0, "end_dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Test case 5: 5D tensor, negative end_dim
    input5 = torch.randn(1, 2, 3, 4, 5).numpy()
    input_dict5 = {"input": input5, "start_dim": 2, "end_dim": -2}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Test case 6: Float tensor
    input6 = torch.randn(2, 3).float().numpy()
    input_dict6 = {"input": input6, "start_dim": 0, "end_dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Test case 7: Int tensor
    input7 = torch.randint(0, 10, (2, 3)).int().numpy()
    input_dict7 = {"input": input7, "start_dim": 0, "end_dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Test case 8: 0D tensor
    input8 = torch.tensor(5).numpy()
    input_dict8 = {"input": input8, "start_dim": 0, "end_dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    # Test case 9: start_dim == end_dim
    input9 = torch.randn(2, 3, 4).numpy()
    input_dict9 = {"input": input9, "start_dim": 1, "end_dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    return list_of_inputs

generated_inputs = flatten_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('flatten', generated_inputs)
