
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy

def enable_onednn_fusion_inputs():
    list_of_inputs = []

    input1 = {"fusion": True}
    list_of_inputs.append(input1)

    input2 = {"fusion": False}
    list_of_inputs.append(input2)

    input3 = {"fusion": True}
    list_of_inputs.append(input3)

    input4 = {"fusion": False}
    list_of_inputs.append(input4)
    
    input5 = {"fusion": True}
    list_of_inputs.append(input5)

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
