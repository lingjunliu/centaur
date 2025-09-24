
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def channel_shuffle_inputs():
    list_of_inputs = []

    # Input 1: Basic 4D tensor with groups = 2
    input1 = torch.randn(1, 4, 5, 5).numpy()
    groups1 = 2
    input_dict1 = {"input": input1, "groups": groups1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs = channel_shuffle_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('channel_shuffle', generated_inputs)
