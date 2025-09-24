
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def iinfo_inputs():
    list_of_inputs = []

    dtypes = [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.float16, torch.float32, torch.float64]

    for dtype in dtypes:
        input_dict = {'dtype': dtype}
        list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = iinfo_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('iinfo', generated_inputs)
