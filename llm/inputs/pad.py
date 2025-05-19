
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def pad_inputs():
    list_of_inputs = []

    # Input 1: 2D float tensor, reflect padding
    input1 = torch.randn(3, 4).numpy()
    pad1 = (1, 1, 2, 2)  # left, right, top, bottom
    mode1 = 'reflect'
    value1 = 0.0
    input_dict1 = {"input": input1, "pad": pad1, "mode": mode1, "value": value1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 3D int tensor, replicate padding
    input2 = torch.randint(0, 10, (2, 3, 5)).numpy()
    pad2 = (0, 0, 1, 1, 2, 2)  # last dim, middle dim, first dim
    mode2 = 'replicate'
    value2 = 0.0
    input_dict2 = {"input": input2, "pad": pad2, "mode": mode2, "value": value2}
    list_of_inputs.append(copy.deepcopy(input_dict2))


    return list_of_inputs

generated_inputs = pad_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('pad', generated_inputs)
