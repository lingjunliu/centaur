
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def spmm_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float32
    input1 = torch.randn(3, 4).to_sparse_coo()
    mat2_1 = torch.randn(4, 5)
    input_dict1 = {"input": input1, "mat2": mat2_1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Different dimensions
    input2 = torch.randn(5, 2).to_sparse_coo()
    mat2_2 = torch.randn(2, 3)
    input_dict2 = {"input": input2, "mat2": mat2_2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Integer sparse tensor and float matrix
    input3 = torch.randint(0, 10, (2, 3)).to_sparse_coo()
    mat2_3 = torch.randn(3, 4)
    input_dict3 = {"input": input3, "mat2": mat2_3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Float sparse tensor and integer matrix
    input4 = torch.randn(4, 5).to_sparse_coo()
    mat2_4 = torch.randint(0, 10, (5, 2))
    input_dict4 = {"input": input4, "mat2": mat2_4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Negative values
    input5 = torch.randn(2, 3).to_sparse_coo()
    mat2_5 = torch.randn(3, 2)
    input_dict5 = {"input": input5, "mat2": mat2_5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = spmm_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('spmm', generated_inputs)
