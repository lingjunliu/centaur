
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

generated_inputs = dict()

import torch
import numpy as np
import copy

def interpolate_inputs():
    list_of_inputs = []

    # Input 1: 4D tensor, size, bilinear, align_corners=False
    input1 = torch.randn(1, 3, 10, 10).numpy()
    size1 = (20, 20)
    input_dict1 = {
        "input": input1,
        "size": size1,
        "scale_factor": None,
        "mode": 'bilinear',
        "align_corners": False,
        "recompute_scale_factor": None,
        "antialias": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 4D tensor, scale_factor, nearest
    input2 = torch.randn(2, 4, 5, 5).numpy()
    scale_factor2 = 2.0
    input_dict2 = {
        "input": input2,
        "size": None,
        "scale_factor": scale_factor2,
        "mode": 'nearest',
        "align_corners": None,
        "recompute_scale_factor": None,
        "antialias": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 5D tensor, size, trilinear, align_corners=True
    input3 = torch.randn(1, 2, 3, 4, 5).numpy()
    size3 = (6, 8, 10)
    input_dict3 = {
        "input": input3,
        "size": size3,
        "scale_factor": None,
        "mode": 'trilinear',
        "align_corners": True,
        "recompute_scale_factor": None,
        "antialias": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 3D tensor, scale_factor tuple, linear, align_corners=False
    input4 = torch.randn(1, 3, 5).numpy()
    scale_factor4 = (2.0,)
    input_dict4 = {
        "input": input4,
        "size": None,
        "scale_factor": scale_factor4,
        "mode": 'linear',
        "align_corners": False,
        "recompute_scale_factor": None,
        "antialias": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 4D tensor, size, bicubic, align_corners=True, recompute_scale_factor=True
    input5 = torch.randn(2, 1, 8, 8).numpy()
    size5 = (16, 16)
    input_dict5 = {
        "input": input5,
        "size": size5,
        "scale_factor": None,
        "mode": 'bicubic',
        "align_corners": True,
        "recompute_scale_factor": False,
        "antialias": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

if __name__ == '__main__':
    generated_inputs = {}
    generated_inputs["torch.nn.functional.interpolate"] = interpolate_inputs()
    #print(generated_inputs)

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('interpolate', generated_inputs['torch.nn.functional.interpolate'], lib="torch")
