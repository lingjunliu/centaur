
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy, numpy as np

def empty_strided_inputs():
    list_of_inputs = []

    input_dict = {
        "size": (2, 3),
        "stride": (3, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "size": (4, 5, 6),
        "stride": (30, 6, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "size": (1, 1, 1, 1),
        "stride": (1, 1, 1, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "size": (7,),
        "stride": (1,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "size": (2, 2, 2, 2, 2),
        "stride": (16, 8, 4, 2, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "size": (10,),
        "stride": (1,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = empty_strided_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('empty_strided', generated_inputs)
