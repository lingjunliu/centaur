
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def unfold_inputs():
    list_of_inputs = []

    # Case 1: Basic case with different kernel_size, stride, padding, and dilation
    input1 = torch.randn(2, 3, 10, 12).numpy()
    kernel_size1 = (3, 4)
    dilation1 = (2, 1)
    padding1 = (1, 0)
    stride1 = (2, 3)
    input_dict1 = {
        "input": input1,
        "kernel_size": kernel_size1,
        "dilation": dilation1,
        "padding": padding1,
        "stride": stride1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    return list_of_inputs

generated_inputs = unfold_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('Unfold', generated_inputs)
