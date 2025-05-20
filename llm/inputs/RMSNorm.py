
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def rmsnorm_inputs():
    list_of_inputs = []

    input_dict = {
        "normalized_shape": [2, 3],
        "eps": 1e-05,
        "elementwise_affine": True,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "normalized_shape": [5],
        "eps": 1e-06,
        "elementwise_affine": False,
        "dtype": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = rmsnorm_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('RMSNorm', generated_inputs)
