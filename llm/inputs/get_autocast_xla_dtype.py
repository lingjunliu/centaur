
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def get_autocast_xla_dtype_inputs():
    list_of_inputs = []

    input1 = {"dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input1))

    input2 = {"dtype": torch.float64}
    list_of_inputs.append(copy.deepcopy(input2))

    input3 = {"dtype": torch.float16}
    list_of_inputs.append(copy.deepcopy(input3))

    input4 = {"dtype": torch.complex64}
    list_of_inputs.append(copy.deepcopy(input4))

    input5 = {"dtype": torch.complex128}
    list_of_inputs.append(copy.deepcopy(input5))

    return list_of_inputs

generated_inputs = get_autocast_xla_dtype_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('get_autocast_xla_dtype', generated_inputs)
