
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy

def enable_onednn_fusion_inputs():
    list_of_inputs = []

    # Test case 1: Enable fusion
    input_dict_1 = {
        "fusion": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Test case 2: Disable fusion
    input_dict_2 = {
        "fusion": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    return list_of_inputs

generated_inputs = enable_onednn_fusion_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('enable_onednn_fusion', generated_inputs)
