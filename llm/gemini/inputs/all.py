
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def all_inputs():
    list_of_inputs = []

    # Input 1: Basic boolean tensor, dim=0, keepdim=False
    input1 = torch.tensor([[True, True], [False, True]]).bool().numpy()
    input_dict1 = {"input": input1, "dim": 0, "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Integer tensor, dim=1, keepdim=True
    input2 = torch.tensor([[1, 1], [0, 1]]).int().numpy()
    input_dict2 = {"input": input2, "dim": 1, "keepdim": True}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Float tensor, dim=0, keepdim=True
    input3 = torch.tensor([[1.0, 1.0], [0.0, 1.0]]).float().numpy()
    input_dict3 = {"input": input3, "dim": 0, "keepdim": True}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 3D boolean tensor, dim=2, keepdim=False
    input4 = torch.tensor([[[True, True], [False, True]], [[True, False], [True, True]]]).bool().numpy()
    input_dict4 = {"input": input4, "dim": 2, "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 2D integer tensor, dim=0, keepdim=False
    input5 = torch.tensor([[1, 1, 1], [1, 1, 1]]).int().numpy()
    input_dict5 = {"input": input5, "dim": 0, "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 2D boolean tensor, dim=1, keepdim=True
    input6 = torch.tensor([[True, True, True], [False, True, False]]).bool().numpy()
    input_dict6 = {"input": input6, "dim": 1, "keepdim": True}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: 3D float tensor, dim=1, keepdim=True
    input7 = torch.randn(2, 3, 4).float().numpy()
    input_dict7 = {"input": input7, "dim": 1, "keepdim": True}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: 1D integer tensor, dim=0 (should raise error due to scalar reduction), keepdim=False
    input8 = torch.tensor([1, 1, 0, 1]).int().numpy()
    input_dict8 = {"input": input8, "dim": 0, "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    # Input 9: 2D float tensor, dim = (0,1), keepdim = True
    input9 = torch.randn(2, 2).float().numpy()
    input_dict9 = {"input": input9, "dim": (0,1), "keepdim": True}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    return list_of_inputs

generated_inputs = all_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('all', generated_inputs)
