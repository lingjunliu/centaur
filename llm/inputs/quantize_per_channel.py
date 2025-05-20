
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def quantize_per_channel_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor, axis=0
    input1 = torch.randn(3, 4, 5).numpy()
    scales1 = torch.rand(3).numpy()
    zero_points1 = np.random.randint(0, 256, size=(3,), dtype=np.int64)
    axis1 = 0
    dtype1 = torch.quint8

    input_dict1 = {
        "input": input1,
        "scales": scales1,
        "zero_points": zero_points1,
        "axis": axis1,
        "dtype": dtype1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    return list_of_inputs

generated_inputs = quantize_per_channel_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('quantize_per_channel', generated_inputs)
