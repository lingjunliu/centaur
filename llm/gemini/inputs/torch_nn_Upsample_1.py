
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def upsample_inputs():
    list_of_inputs = []

    # Input 1: Nearest neighbor, 2D input
    input1 = torch.randn(1, 3, 16, 16).numpy()
    input_dict1 = {
        "size": 32,
        "scale_factor": None,
        "mode": "nearest",
        "align_corners": None,
        "recompute_scale_factor": None,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Bilinear, 2D input, align_corners=False
    input2 = torch.randn(2, 1, 8, 8).numpy()
    input_dict2 = {
        "size": 16,
        "scale_factor": None,
        "mode": "bilinear",
        "align_corners": False,
        "recompute_scale_factor": None,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Bicubic, 2D input, align_corners=True
    input3 = torch.randn(1, 4, 4, 4).numpy()
    input_dict3 = {
        "size": 8,
        "scale_factor": None,
        "mode": "bicubic",
        "align_corners": True,
        "recompute_scale_factor": None,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Linear, 1D input, scale_factor
    input4 = torch.randn(1, 2, 10).numpy()
    input_dict4 = {
        "size": None,
        "scale_factor": 2.0,
        "mode": "linear",
        "align_corners": None,
        "recompute_scale_factor": None,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Trilinear, 3D input, scale_factor
    input5 = torch.randn(1, 1, 4, 4, 4).numpy()
    input_dict5 = {
        "size": None,
        "scale_factor": 2.0,
        "mode": "trilinear",
        "align_corners": False,
        "recompute_scale_factor": None,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: Tuple size, 2D input, bilinear
    input6 = torch.randn(1, 3, 10, 12).numpy()
    input_dict6 = {
        "size": (20, 24),
        "scale_factor": None,
        "mode": "bilinear",
        "align_corners": False,
        "recompute_scale_factor": None,
        "input": input6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Tuple size, 3D input, trilinear
    input7 = torch.randn(1, 1, 5, 6, 7).numpy()
    input_dict7 = {
        "size": (10, 12, 14),
        "scale_factor": None,
        "mode": "trilinear",
        "align_corners": True,
        "recompute_scale_factor": None,
        "input": input7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Tuple scale factor, 2D input, bilinear
    input8 = torch.randn(1, 3, 10, 12).numpy()
    input_dict8 = {
        "size": None,
        "scale_factor": (2.0, 2.0),
        "mode": "bilinear",
        "align_corners": False,
        "recompute_scale_factor": None,
        "input": input8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    # Input 9: Recompute scale factor, bilinear
    input9 = torch.randn(1, 1, 8, 8).numpy()
    input_dict9 = {
        "size": None,
        "scale_factor": 2.0,
        "mode": "bilinear",
        "align_corners": False,
        "recompute_scale_factor": True,
        "input": input9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: 3D Input, Nearest Neighbor
    input10 = torch.randn(1, 2, 5, 5, 5).numpy()
    input_dict10 = {
        "size": 10,
        "scale_factor": None,
        "mode": "nearest",
        "align_corners": None,
        "recompute_scale_factor": None,
        "input": input10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.Upsample_1"] = upsample_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.Upsample_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Upsample_1'.")

check_valid('torch.nn.Upsample', generated_inputs['torch.nn.Upsample_1'], lib="torch", suffix=1)
