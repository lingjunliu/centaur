
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def sym_float_inputs():
    generated_inputs = []

    # Input 1: Scalar float
    a1 = np.array(3.14, dtype=np.float32)
    generated_inputs.append({"a": a1})

    return generated_inputs

generated_inputs = sym_float_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('sym_float', generated_inputs)
