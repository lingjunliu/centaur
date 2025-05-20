
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def BatchNorm1d_inputs():
    list_of_inputs = []

    input_dict = {
        "num_features": 100,
        "eps": 1e-05,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "input": np.random.randn(20, 100)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_features": 50,
        "eps": 1e-03,
        "momentum": 0.2,
        "affine": False,
        "track_running_stats": False,
        "input": np.random.randn(10, 50)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_features": 256,
        "eps": 1e-08,
        "momentum": 0.05,
        "affine": True,
        "track_running_stats": True,
        "input": np.random.randn(5, 256, 5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_features": 128,
        "eps": 1e-06,
        "momentum": 0.9,
        "affine": False,
        "track_running_stats": False,
        "input": np.random.randn(30, 128)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_features": 64,
        "eps": 1e-07,
        "momentum": None,
        "affine": True,
        "track_running_stats": True,
        "input": np.random.randn(2, 64, 10)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = BatchNorm1d_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('BatchNorm1d', generated_inputs)
