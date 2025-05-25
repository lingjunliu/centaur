
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy

def avg_pool1d_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float tensor
    input1 = torch.randn(1, 3, 10).numpy()
    input_dict1 = {
        "input": input1,
        "kernel_size": 3,
        "stride": 2,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Integer tensor
    input2 = torch.randint(0, 10, (1, 4, 15)).numpy()
    input_dict2 = {
        "input": input2,
        "kernel_size": 4,
        "stride": 3,
        "padding": 1,
        "ceil_mode": True,
        "count_include_pad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    return list_of_inputs

generated_inputs = avg_pool1d_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('avg_pool1d', generated_inputs)
