
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np
import torch.nn as nn

def torch_jit_script_inputs():
    generated_inputs = []

    # 1. Scripting a simple function
    def simple_func(x, y):
        if x.max() > y.max():
            r = x
        else:
            r = y
        return r

    input_dict = {
        'obj': simple_func,
        'optimize': True,
        '_frames_up': 0,
        '_rcb': None,
        'example_inputs': [(torch.ones(2, 2).numpy(), torch.ones(2, 2).numpy())]
    }
    generated_inputs.append(copy.deepcopy(input_dict))

    return generated_inputs

generated_inputs = torch_jit_script_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('script', generated_inputs)
