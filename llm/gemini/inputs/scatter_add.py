
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def scatter_add_inputs():
    list_of_inputs = []

    # Case 1: Basic case with float tensor, 1D
    input1 = torch.zeros(5).float().numpy()
    index1 = torch.tensor([0, 1, 2, 0, 3]).long().numpy()
    src1 = torch.randn(5).float().numpy()
    input_dict1 = {"input": input1, "dim": 0, "index": index1, "src": src1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: 2D input, dim=0. Reduced index range to avoid out-of-bounds error.
    input2 = torch.zeros(3, 5).float().numpy()
    index2 = torch.tensor([[0, 1, 2, 0, 1], [1, 2, 0, 2, 0]]).long().numpy()
    src2 = torch.randn(2, 5).float().numpy()
    input_dict2 = {"input": input2, "dim": 0, "index": index2, "src": src2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: 2D input, dim=1. Reduced index range to avoid out-of-bounds error.
    input3 = torch.zeros(2, 4).float().numpy()
    index3 = torch.tensor([[0, 1, 2, 3], [2, 3, 0, 1]]).long().numpy()
    src3 = torch.randn(2, 4).float().numpy()
    input_dict3 = {"input": input3, "dim": 1, "index": index3, "src": src3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: Integer input. Reduced index range to avoid out-of-bounds error.
    input4 = torch.zeros(5).int().numpy()
    index4 = torch.tensor([0, 1, 2, 0, 3]).long().numpy()
    src4 = torch.randint(0, 10, (5,)).int().numpy()
    input_dict4 = {"input": input4, "dim": 0, "index": index4, "src": src4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Case 5: 3D input, dim=0. Reduced index range to avoid out-of-bounds error.
    input5 = torch.zeros(3, 2, 4).float().numpy()
    index5 = torch.tensor([[[0, 1, 2, 0], [1, 0, 2, 2]], [[1, 2, 0, 1], [0, 2, 1, 2]]]).long().numpy()
    src5 = torch.randn(2, 2, 4).float().numpy()
    input_dict5 = {"input": input5, "dim": 0, "index": index5, "src": src5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = scatter_add_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('scatter_add', generated_inputs)
