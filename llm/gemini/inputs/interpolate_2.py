
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def interpolate_inputs():
    list_of_inputs = []

    # Input 1: Basic 4D tensor with size
    input = torch.randn(1, 3, 16, 16).numpy()
    size = (32, 32)
    input_dict = {
        "input": input,
        "size": size,
        "scale_factor": None,
        "mode": 'bilinear',
        "align_corners": False,
        "recompute_scale_factor": None,
        "antialias": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic 4D tensor with scale_factor
    input = torch.randn(1, 3, 16, 16).numpy()
    scale_factor = 2.0
    input_dict = {
        "input": input,
        "size": None,
        "scale_factor": scale_factor,
        "mode": 'nearest',
        "align_corners": None,
        "recompute_scale_factor": None,
        "antialias": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 5D tensor with size, trilinear
    input = torch.randn(1, 3, 8, 16, 16).numpy()
    size = (16, 32, 32)
    input_dict = {
        "input": input,
        "size": size,
        "scale_factor": None,
        "mode": 'trilinear',
        "align_corners": False,
        "recompute_scale_factor": None,
        "antialias": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D tensor with scale_factor, linear
    input = torch.randn(1, 3, 16).numpy()
    scale_factor = 2.0
    input_dict = {
        "input": input,
        "size": None,
        "scale_factor": scale_factor,
        "mode": 'linear',
        "align_corners": None,
        "recompute_scale_factor": None,
        "antialias": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D tensor with size, align_corners=True
    input = torch.randn(1, 3, 16, 16).numpy()
    size = (32, 32)
    input_dict = {
        "input": input,
        "size": size,
        "scale_factor": None,
        "mode": 'bilinear',
        "align_corners": True,
        "recompute_scale_factor": None,
        "antialias": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D tensor with scale_factor tuple
    input = torch.randn(1, 3, 16, 16).numpy()
    scale_factor = (2.0, 2.0)
    input_dict = {
        "input": input,
        "size": None,
        "scale_factor": scale_factor,
        "mode": 'bilinear',
        "align_corners": False,
        "recompute_scale_factor": None,
        "antialias": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: area mode
    input = torch.randn(1, 3, 16, 16).numpy()
    size = (8, 8)
    input_dict = {
        "input": input,
        "size": size,
        "scale_factor": None,
        "mode": 'area',
        "align_corners": None,
        "recompute_scale_factor": None,
        "antialias": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: nearest-exact mode
    input = torch.randn(1, 3, 16, 16).numpy()
    size = (8, 8)
    input_dict = {
        "input": input,
        "size": size,
        "scale_factor": None,
        "mode": 'nearest-exact',
        "align_corners": None,
        "recompute_scale_factor": None,
        "antialias": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: bicubic mode with antialias
    input = torch.randn(1, 3, 16, 16).numpy()
    size = (32, 32)
    input_dict = {
        "input": input,
        "size": size,
        "scale_factor": None,
        "mode": 'bicubic',
        "align_corners": False,
        "recompute_scale_factor": None,
        "antialias": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.interpolate_2"] = interpolate_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.interpolate_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.interpolate_2'.")

check_valid('torch.nn.functional.interpolate', generated_inputs['torch.nn.functional.interpolate_2'], lib="torch")
