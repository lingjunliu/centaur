
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def set_default_dtype_inputs():
    list_of_inputs = []

    input_dict = {"d": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"d": torch.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"d": torch.float16}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"d": torch.bfloat16}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = set_default_dtype_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('set_default_dtype', generated_inputs)
