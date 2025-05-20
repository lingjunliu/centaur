
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def torch_jit_script_inputs():
    generated_inputs = []

    def test_sum(a, b):
        return a + b

    example_inputs = [(np.array(3), np.array(4))]
    input_dict = {
        'obj': test_sum,
        'optimize': True,
        '_frames_up': 0,
        '_rcb': None,
        'example_inputs': example_inputs
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
