
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def gradient_inputs():
    list_of_inputs = []

    # Case 1: Basic 1D tensor with default spacing and edge_order
    input_1 = torch.tensor([1.0, 2.0, 4.0, 7.0, 11.0]).numpy()
    input_dict_1 = {
        "input": input_1,
        "dim": None,
        "spacing": None,
        "edge_order": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    return list_of_inputs

generated_inputs = gradient_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('gradient', generated_inputs)
