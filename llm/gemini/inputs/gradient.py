
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def gradient_inputs():
    list_of_inputs = []

    # Case 1: Basic 1D tensor, default spacing and edge_order
    input_1 = torch.tensor([1.0, 2.0, 4.0, 7.0, 11.0]).numpy()
    input_dict_1 = {
        "input": input_1,
        "dim": None,
        "spacing": None,
        "edge_order": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: 2D tensor, scalar spacing
    input_2 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32).numpy()
    input_dict_2 = {
        "input": input_2,
        "dim": None,
        "spacing": 2.0,
        "edge_order": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: 2D tensor, list of scalar spacing, specific dim
    input_3 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32).numpy()
    input_dict_3 = {
        "input": input_3,
        "dim": (1,),
        "spacing": (2.0,),
        "edge_order": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

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
