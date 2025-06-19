
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def interpolate_inputs():
    list_of_inputs = []

    # Input 1: Basic 4D tensor, scale_factor, bilinear, align_corners=False
    input_tensor = torch.randn(1, 3, 10, 10).numpy()
    input_dict = {
        "input": input_tensor,
        "size": None,
        "scale_factor": 2.0,
        "mode": 'bilinear',
        "align_corners": False,
        "recompute_scale_factor": None,
        "antialias": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 5D tensor, size, trilinear, align_corners=True
    input_tensor = torch.randn(2, 4, 5, 8, 12).numpy()
    input_dict = {
        "input": input_tensor,
        "size": (10, 16, 24),
        "scale_factor": None,
        "mode": 'trilinear',
        "align_corners": True,
        "recompute_scale_factor": None,
        "antialias": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor, scale_factor tuple, linear, align_corners=False
    input_tensor = torch.randn(1, 2, 15).numpy()
    input_dict = {
        "input": input_tensor,
        "size": None,
        "scale_factor": (2.5,),
        "mode": 'linear',
        "align_corners": False,
        "recompute_scale_factor": None,
        "antialias": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D tensor, size tuple, bicubic, align_corners=True, antialias=True
    input_tensor = torch.randn(1, 1, 7, 7).numpy()
    input_dict = {
        "input": input_tensor,
        "size": (14, 14),
        "scale_factor": None,
        "mode": 'bicubic',
        "align_corners": True,
        "recompute_scale_factor": None,
        "antialias": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 4D tensor, scale_factor, nearest
    input_tensor = torch.randn(1, 1, 5, 5).numpy()
    input_dict = {
        "input": input_tensor,
        "size": None,
        "scale_factor": 3.0,
        "mode": 'nearest',
        "align_corners": None,
        "recompute_scale_factor": None,
        "antialias": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.interpolate_1"] = interpolate_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.interpolate_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.interpolate_1'.")

check_valid('torch.nn.functional.interpolate', generated_inputs['torch.nn.functional.interpolate_1'], lib="torch")
