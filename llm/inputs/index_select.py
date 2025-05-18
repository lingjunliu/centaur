
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def index_select_inputs():
    list_of_inputs = []

    input1 = torch.randn(3, 4, 5).numpy()
    dim1 = 0
    index1 = torch.tensor([0, 2]).long().numpy()
    input_dict1 = {
        "input": input1,
        "dim": dim1,
        "index": index1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 2).numpy()
    dim2 = 1
    index2 = torch.tensor([0]).long().numpy()
    input_dict2 = {
        "input": input2,
        "dim": dim2,
        "index": index2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(5, 3).numpy()
    dim3 = 0
    index3 = torch.tensor([1, 4, 2]).long().numpy()
    input_dict3 = {
        "input": input3,
        "dim": dim3,
        "index": index3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(2, 3, 4).numpy()
    dim4 = 1
    index4 = torch.tensor([0, 2]).long().numpy()
    input_dict4 = {
        "input": input4,
        "dim": dim4,
        "index": index4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(4, 4).numpy()
    dim5 = 0
    index5 = torch.tensor([3, 1, 0]).long().numpy()
    input_dict5 = {
        "input": input5,
        "dim": dim5,
        "index": index5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

list_of_inputs = index_select_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('index_select', list_of_inputs)
