
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def ndtr_inputs():
    list_of_inputs = []

    x1 = np.array([0.0, 1.0, -1.0, 2.0, -2.0], dtype=np.float32)
    input_dict1 = {"x": x1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs = ndtr_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('ndtr', generated_inputs)
