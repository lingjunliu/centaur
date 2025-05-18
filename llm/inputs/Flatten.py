
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def flatten_inputs():
    list_of_inputs = []

    input1 = torch.randn(1, 3, 5, 5).numpy()
    start_dim1 = 1
    end_dim1 = 3
    input_dict1 = {
        "input": input1,
        "start_dim": start_dim1,
        "end_dim": end_dim1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 4, 6).numpy()
    start_dim2 = 0
    end_dim2 = 1
    input_dict2 = {
        "input": input2,
        "start_dim": start_dim2,
        "end_dim": end_dim2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(10).numpy()
    start_dim3 = 0
    end_dim3 = 0
    input_dict3 = {
        "input": input3,
        "start_dim": start_dim3,
        "end_dim": end_dim3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(2, 3, 4, 5).numpy()
    start_dim4 = 2
    end_dim4 = 2
    input_dict4 = {
        "input": input4,
        "start_dim": start_dim4,
        "end_dim": end_dim4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randn(1, 2, 3).numpy()
    start_dim5 = 0
    end_dim5 = 2
    input_dict5 = {
        "input": input5,
        "start_dim": start_dim5,
        "end_dim": end_dim5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    

    return list_of_inputs

list_of_inputs = flatten_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('Flatten', list_of_inputs)
