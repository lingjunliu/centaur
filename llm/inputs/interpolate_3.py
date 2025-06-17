
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def interpolate_inputs():
    list_of_inputs = []

    # Input 1: 4D tensor, size, bilinear, align_corners=False
    input1 = torch.randn(1, 3, 16, 16).numpy()
    size1 = (32, 32)
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

    # Input 2: 5D tensor, scale_factor, trilinear, align_corners=True
    input2 = torch.randn(2, 4, 8, 8, 8).numpy()
    scale_factor2 = (2.0, 2.0, 2.0)
    input_dict2 = {
        "input": input2,
        "size": None,
        "scale_factor": scale_factor2,
        "mode": 'trilinear',
        "align_corners": True,
        "recompute_scale_factor": None,
        "antialias": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D tensor, size, linear, align_corners=False (doesn't matter for linear)
    input3 = torch.randn(1, 2, 32).numpy()
    size3 = (64,)
    input_dict3 = {
        "input": input3,
        "size": size3,
        "scale_factor": None,
        "mode": 'linear',
        "align_corners": False,
        "recompute_scale_factor": None,
        "antialias": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 4D tensor, scale_factor, nearest
    input4 = torch.randn(1, 1, 10, 10).numpy()
    scale_factor4 = (3.0, 3.0)
    input_dict4 = {
        "input": input4,
        "size": None,
        "scale_factor": scale_factor4,
        "mode": 'nearest',
        "align_corners": None,
        "recompute_scale_factor": None,
        "antialias": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: 4D tensor, size, bicubic, align_corners=True
    input5 = torch.randn(1, 3, 16, 16).numpy()
    size5 = (32, 32)
    input_dict5 = {
        "input": input5,
        "size": size5,
        "scale_factor": None,
        "mode": 'bicubic',
        "align_corners": True,
        "recompute_scale_factor": None,
        "antialias": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.interpolate_3"] = interpolate_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.interpolate_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.interpolate_3'.")

check_valid('torch.nn.functional.interpolate', generated_inputs['torch.nn.functional.interpolate_3'], lib="torch")
