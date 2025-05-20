
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def unfold_inputs():
    list_of_inputs = []

    # Test case 1: Basic case with different kernel sizes
    input = torch.randn(2, 3, 10, 12).numpy()
    kernel_size = (3, 4)
    dilation = (1, 1)
    padding = (0, 0)
    stride = (1, 1)
    input_dict = {'input': input, 'kernel_size': kernel_size, 'dilation': dilation, 'padding': padding, 'stride': stride}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
