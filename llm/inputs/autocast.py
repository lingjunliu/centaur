
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def autocast_inputs():
    list_of_inputs = []

    input1 = {
        "device_type": "cuda",
        "enabled": True
    }
    list_of_inputs.append(copy.deepcopy(input1))

    input2 = {
        "device_type": "cpu",
        "enabled": False
    }
    list_of_inputs.append(copy.deepcopy(input2))

    return list_of_inputs

generated_inputs = autocast_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('autocast', generated_inputs)
