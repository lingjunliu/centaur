
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def avgpool1d_inputs():
    list_of_inputs = []

    input1 = torch.randn(1, 1, 7).numpy()
    input_dict1 = {
        'input': input1,
        'kernel_size': 3,
        'stride': 2,
        'padding': 0,
        'ceil_mode': False,
        'count_include_pad': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(1, 2, 10).numpy()
    input_dict2 = {
        'input': input2,
        'kernel_size': 2,
        'stride': 1,
        'padding': 1,
        'ceil_mode': False,
        'count_include_pad': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(2, 3, 15).numpy()
    input_dict3 = {
        'input': input3,
        'kernel_size': 4,
        'stride': 3,
        'padding': 2,
        'ceil_mode': True,
        'count_include_pad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(1, 10).numpy()
    input_dict4 = {
        'input': input4,
        'kernel_size': 3,
        'stride': 1,
        'padding': 0,
        'ceil_mode': False,
        'count_include_pad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    return list_of_inputs

generated_inputs = avgpool1d_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('AvgPool1d', generated_inputs)
