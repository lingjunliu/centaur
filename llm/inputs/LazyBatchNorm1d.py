
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def lazy_batch_norm1d_inputs():
    list_of_inputs = []

    input_dict1 = {
        "eps": 1e-05,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input_dict2 = {
        "eps": 1e-03,
        "momentum": 0.2,
        "affine": False,
        "track_running_stats": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input_dict3 = {
        "eps": 1e-07,
        "momentum": 0.05,
        "affine": True,
        "track_running_stats": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    return list_of_inputs

generated_inputs = lazy_batch_norm1d_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('LazyBatchNorm1d', generated_inputs)
