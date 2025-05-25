
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def rrelu_inputs():
    list_of_inputs = []

    input_dict_1 = {
        "lower": 0.1,
        "upper": 0.3,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))
    
    return list_of_inputs

generated_inputs = rrelu_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('RReLU', generated_inputs)
