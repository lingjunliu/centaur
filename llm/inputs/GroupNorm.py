
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def groupnorm_inputs():
    list_of_inputs = []

    input = torch.randn(2, 6, 5, 5).numpy()
    num_groups = 3
    num_channels = 6
    eps = 1e-5
    affine = True
    input_dict = {
        "input": input,
        "num_groups": num_groups,
        "num_channels": num_channels,
        "eps": eps,
        "affine": affine
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 12, 10, 10).numpy()
    num_groups = 4
    num_channels = 12
    eps = 1e-8
    affine = False
    input_dict = {
        "input": input,
        "num_groups": num_groups,
        "num_channels": num_channels,
        "eps": eps,
        "affine": affine
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(4, 8, 8, 8).numpy()
    num_groups = 2
    num_channels = 8
    eps = 1e-3
    affine = True
    input_dict = {
        "input": input,
        "num_groups": num_groups,
        "num_channels": num_channels,
        "eps": eps,
        "affine": affine
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 16, 12, 12).numpy()
    num_groups = 8
    num_channels = 16
    eps = 1e-6
    affine = False
    input_dict = {
        "input": input,
        "num_groups": num_groups,
        "num_channels": num_channels,
        "eps": eps,
        "affine": affine
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(3, 4, 7, 7).numpy()
    num_groups = 1
    num_channels = 4
    eps = 1e-7
    affine = True
    input_dict = {
        "input": input,
        "num_groups": num_groups,
        "num_channels": num_channels,
        "eps": eps,
        "affine": affine
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = groupnorm_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('GroupNorm', list_of_inputs)
