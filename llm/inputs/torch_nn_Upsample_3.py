
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def upsample_inputs():
    list_of_inputs = []

    # Input 1: 4D tensor, scale_factor, nearest
    input1 = np.arange(1, 17, dtype=np.float32).reshape((1, 1, 4, 4))
    input_dict1 = {
        "size": None,
        "scale_factor": (2.0, 2.0),
        "mode": "nearest",
        "align_corners": False,
        "recompute_scale_factor": False,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 4D tensor, size, bilinear, align_corners=False
    input2 = np.arange(1, 10, dtype=np.float32).reshape((1, 1, 3, 3))
    input_dict2 = {
        "size": (6, 6),
        "scale_factor": None,
        "mode": "bilinear",
        "align_corners": False,
        "recompute_scale_factor": False,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 4D tensor, scale_factor, bilinear, align_corners=True
    input3 = np.arange(1, 10, dtype=np.float32).reshape((1, 1, 3, 3))
    input_dict3 = {
        "size": None,
        "scale_factor": (2.0, 2.0),
        "mode": "bilinear",
        "align_corners": True,
        "recompute_scale_factor": False,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 5D tensor, scale_factor, trilinear, align_corners=False
    input4 = np.arange(1, 28, dtype=np.float32).reshape((1, 1, 3, 3, 3))
    input_dict4 = {
        "size": None,
        "scale_factor": (2.0, 2.0, 2.0),
        "mode": "trilinear",
        "align_corners": False,
        "recompute_scale_factor": False,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 5D tensor, size, trilinear, align_corners=True
    input5 = np.arange(1, 28, dtype=np.float32).reshape((1, 1, 3, 3, 3))
    input_dict5 = {
        "size": (6, 6, 6),
        "scale_factor": None,
        "mode": "trilinear",
        "align_corners": True,
        "recompute_scale_factor": False,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 3D tensor, scale_factor, linear
    input6 = np.arange(1, 10, dtype=np.float32).reshape((1, 1, 9))
    input_dict6 = {
        "size": None,
        "scale_factor": (2.0,),
        "mode": "linear",
        "align_corners": True,
        "recompute_scale_factor": False,
        "input": input6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.Upsample_3"] = upsample_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.Upsample_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Upsample_3'.")

check_valid('torch.nn.Upsample', generated_inputs['torch.nn.Upsample_3'], lib="torch", suffix=3)
