
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def generate_upsample_inputs():
    list_of_inputs = []

    # Input 1: scale_factor, mode='nearest'
    input_tensor = np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2)
    input_dict = {
        "size": None,
        "scale_factor": 2.0,
        "mode": 'nearest',
        "align_corners": False,
        "recompute_scale_factor": False,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: size, mode='bilinear', align_corners=False
    input_tensor = np.arange(1, 10, dtype=np.float32).reshape(1, 1, 3, 3)
    input_dict = {
        "size": (6, 6),
        "scale_factor": None,
        "mode": 'bilinear',
        "align_corners": False,
        "recompute_scale_factor": False,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: scale_factor, mode='bilinear', align_corners=True
    input_tensor = np.arange(1, 17, dtype=np.float32).reshape(1, 1, 4, 4)
    input_dict = {
        "size": None,
        "scale_factor": 1.5,
        "mode": 'bilinear',
        "align_corners": True,
        "recompute_scale_factor": False,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: size, mode='trilinear'
    input_tensor = np.arange(1, 28, dtype=np.float32).reshape(1, 1, 3, 3, 3)
    input_dict = {
        "size": (6, 6, 6),
        "scale_factor": None,
        "mode": 'trilinear',
        "align_corners": False,
        "recompute_scale_factor": False,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: scale_factor, mode='linear'
    input_tensor = np.arange(1, 11, dtype=np.float32).reshape(1, 1, 10)
    input_dict = {
        "size": None,
        "scale_factor": 2.5,
        "mode": 'linear',
        "align_corners": False,
        "recompute_scale_factor": False,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: size, mode='bicubic', align_corners=True
    input_tensor = np.arange(1, 26, dtype=np.float32).reshape(1, 1, 5, 5)
    input_dict = {
        "size": (10, 10),
        "scale_factor": None,
        "mode": 'bicubic',
        "align_corners": True,
        "recompute_scale_factor": False,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: scale_factor tuple, mode='bilinear'
    input_tensor = np.arange(1, 10, dtype=np.float32).reshape(1, 1, 3, 3)
    input_dict = {
        "size": None,
        "scale_factor": (2.0, 2.0),
        "mode": 'bilinear',
        "align_corners": False,
        "recompute_scale_factor": False,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.Upsample_2"] = generate_upsample_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.Upsample_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Upsample_2'.")

check_valid('torch.nn.Upsample', generated_inputs['torch.nn.Upsample_2'], lib="torch", suffix=2)
