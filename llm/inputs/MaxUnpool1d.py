
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def max_unpool1d_inputs():
    list_of_inputs = []

    # Case 1: Basic case with N, C, Hin
    input_tensor = torch.randn(1, 1, 4)
    pool = torch.nn.MaxPool1d(kernel_size=2, stride=2, return_indices=True)
    output_tensor, indices_tensor = pool(input_tensor)
    input_dict = {
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "input": output_tensor.numpy(),
        "indices": indices_tensor.numpy(),
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = max_unpool1d_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('MaxUnpool1d', generated_inputs)
