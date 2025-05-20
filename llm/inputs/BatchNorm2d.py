
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def batchnorm2d_inputs():
    list_of_inputs = []

    input_dict = {
        "num_features": 100,
        "eps": 1e-05,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_features": 50,
        "eps": 1e-04,
        "momentum": 0.2,
        "affine": False,
        "track_running_stats": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_features": 256,
        "eps": 1e-08,
        "momentum": 0.05,
        "affine": True,
        "track_running_stats": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_features": 128,
        "eps": 1e-06,
        "momentum": 0.15,
        "affine": False,
        "track_running_stats": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_features": 64,
        "eps": 1e-07,
        "momentum": None,
        "affine": True,
        "track_running_stats": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = batchnorm2d_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('BatchNorm2d', generated_inputs)
