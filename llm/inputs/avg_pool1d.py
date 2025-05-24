
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def avg_pool1d_inputs():
    list_of_inputs = []

    input1 = torch.randn(1, 3, 10).numpy()
    kernel_size1 = 3
    stride1 = 2
    padding1 = 1
    ceil_mode1 = False
    count_include_pad1 = True
    input_dict1 = {
        "input": input1,
        "kernel_size": kernel_size1,
        "stride": stride1,
        "padding": padding1,
        "ceil_mode": ceil_mode1,
        "count_include_pad": count_include_pad1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 3, 15).numpy()
    kernel_size2 = 4
    stride2 = 3
    padding2 = 0
    ceil_mode2 = True
    count_include_pad2 = False
    input_dict2 = {
        "input": input2,
        "kernel_size": kernel_size2,
        "stride": stride2,
        "padding": padding2,
        "ceil_mode": ceil_mode2,
        "count_include_pad": count_include_pad2
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
