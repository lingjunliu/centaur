
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy as np

def upsample_inputs():
    list_of_inputs = []

    # Input 1: 4D (Image) Upsampling with scale_factor and 'nearest' mode.
    # Use recompute_scale_factor=True to indicate scale_factor should be used.
    input_1 = {
        'size': (1, 1),
        'scale_factor': (2.0, 2.0),
        'mode': 'nearest',
        'align_corners': False,
        'recompute_scale_factor': True,
        'input': torch.randn(1, 3, 16, 16).numpy(),
    }
    list_of_inputs.append(copy.deepcopy(input_1))

    # Input 2: 4D Upsampling with target 'size' and 'bilinear' mode
    # Use recompute_scale_factor=False to indicate size should be used.
    input_2 = {
        'size': (20, 20),
        'scale_factor': (1.0, 1.0),
        'mode': 'bilinear',
        'align_corners': False,
        'recompute_scale_factor': False,
        'input': torch.randn(1, 1, 10, 10).numpy(),
    }
    list_of_inputs.append(copy.deepcopy(input_2))

    # Input 3: 4D Bilinear with align_corners=True and non-square input
    input_3 = {
        'size': (1, 1),
        'scale_factor': (2.0, 2.0),
        'mode': 'bilinear',
        'align_corners': True,
        'recompute_scale_factor': True,
        'input': torch.randn(2, 1, 8, 12).numpy(),
    }
    list_of_inputs.append(copy.deepcopy(input_3))

    # Input 4: 4D Bicubic Upsampling with floating point scale_factor
    input_4 = {
        'size': (1, 1),
        'scale_factor': (1.5, 1.5),
        'mode': 'bicubic',
        'align_corners': False,
        'recompute_scale_factor': True,
        'input': torch.randn(1, 3, 8, 8).numpy(),
    }
    list_of_inputs.append(copy.deepcopy(input_4))

    # Input 5: 3D (Temporal) Upsampling with 'linear' mode
    input_5 = {
        'size': (1,),
        'scale_factor': (3.0,),
        'mode': 'linear',
        'align_corners': False,
        'recompute_scale_factor': True,
        'input': torch.randn(2, 8, 32).numpy(),
    }
    list_of_inputs.append(copy.deepcopy(input_5))

    # Input 6: 3D Upsampling with target 'size' and align_corners=True
    input_6 = {
        'size': (45,),
        'scale_factor': (1.0,),
        'mode': 'linear',
        'align_corners': True,
        'recompute_scale_factor': False,
        'input': torch.randn(4, 5, 15).numpy(),
    }
    list_of_inputs.append(copy.deepcopy(input_6))

    # Input 7: 5D (Volumetric) Upsampling with 'trilinear' mode
    input_7 = {
        'size': (1, 1, 1),
        'scale_factor': (2.0, 2.0, 2.0),
        'mode': 'trilinear',
        'align_corners': False,
        'recompute_scale_factor': True,
        'input': torch.randn(1, 1, 4, 8, 8).numpy(),
    }
    list_of_inputs.append(copy.deepcopy(input_7))

    # Input 8: 5D Upsampling with target 'size' and align_corners=True
    input_8 = {
        'size': (10, 12, 14),
        'scale_factor': (1.0, 1.0, 1.0),
        'mode': 'trilinear',
        'align_corners': True,
        'recompute_scale_factor': False,
        'input': torch.randn(2, 3, 5, 6, 7).numpy(),
    }
    list_of_inputs.append(copy.deepcopy(input_8))

    # Input 9: Using recompute_scale_factor=True explicitly
    input_9 = {
        'size': (1, 1),
        'scale_factor': (2.0, 2.0),
        'mode': 'bilinear',
        'align_corners': False,
        'recompute_scale_factor': True,
        'input': torch.randn(1, 1, 7, 7).numpy(),
    }
    list_of_inputs.append(copy.deepcopy(input_9))

    # Input 10: Non-uniform scaling for 4D input
    input_10 = {
        'size': (1, 1),
        'scale_factor': (2.0, 1.5),
        'mode': 'bilinear',
        'align_corners': False,
        'recompute_scale_factor': True,
        'input': torch.randn(1, 3, 4, 6).numpy(),
    }
    list_of_inputs.append(copy.deepcopy(input_10))

    return list_of_inputs

generated_inputs["torch.nn.Upsample_3"] = upsample_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Upsample_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Upsample_3'.")

check_valid('torch.nn.Upsample', generated_inputs['torch.nn.Upsample_3'], lib="torch", suffix=3)
