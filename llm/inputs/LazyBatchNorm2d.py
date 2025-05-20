
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def lazy_batchnorm2d_inputs():
    list_of_inputs = []

    input_dict = {
        "eps": 1e-05,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "dtype": torch.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "eps": 1e-03,
        "momentum": 0.2,
        "affine": False,
        "track_running_stats": False,
        "dtype": torch.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "eps": 1e-07,
        "momentum": None,
        "affine": True,
        "track_running_stats": True,
        "dtype": torch.float16
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "eps": 0.0,
        "momentum": 0.99,
        "affine": False,
        "track_running_stats": False,
        "dtype": torch.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = lazy_batchnorm2d_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('LazyBatchNorm2d', generated_inputs)
