
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def fake_quantize_per_channel_affine_inputs():
    list_of_inputs = []

    # Input 1: Basic case, 1D input
    input1 = torch.tensor([-1.0, 0.0, 1.0, 2.0]).numpy()
    scale1 = torch.tensor([0.1, 0.2, 0.3, 0.4]).numpy()
    zero_point1 = torch.tensor([0, 1, 2, 3], dtype=torch.int32).numpy()
    quant_min1 = 0
    quant_max1 = 255
    ch_axis1 = 0

    input_dict1 = {
        "input": input1,
        "scale": scale1,
        "zero_point": zero_point1,
        "quant_min": quant_min1,
        "quant_max": quant_max1,
        "ch_axis": ch_axis1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D input
    input2 = torch.tensor([[-1.0, 0.0], [1.0, 2.0]]).numpy()
    scale2 = torch.tensor([0.1, 0.2]).numpy()
    zero_point2 = torch.tensor([0, 1], dtype=torch.int32).numpy()
    quant_min2 = 0
    quant_max2 = 255
    ch_axis2 = 1

    input_dict2 = {
        "input": input2,
        "scale": scale2,
        "zero_point": zero_point2,
        "quant_min": quant_min2,
        "quant_max": quant_max2,
        "ch_axis": ch_axis2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D input
    input3 = torch.randn(2, 3, 4).numpy()
    scale3 = torch.abs(torch.randn(3)).numpy()
    zero_point3 = torch.randint(0, 256, (3,), dtype=torch.int32).numpy()
    quant_min3 = 0
    quant_max3 = 255
    ch_axis3 = 1

    input_dict3 = {
        "input": input3,
        "scale": scale3,
        "zero_point": zero_point3,
        "quant_min": quant_min3,
        "quant_max": quant_max3,
        "ch_axis": ch_axis3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Different quant_min and quant_max
    input4 = torch.tensor([-1.0, 0.0, 1.0, 2.0]).numpy()
    scale4 = torch.tensor([0.1, 0.2, 0.3, 0.4]).numpy()
    zero_point4 = torch.tensor([128, 129, 130, 131], dtype=torch.int32).numpy()
    quant_min4 = 0
    quant_max4 = 255
    ch_axis4 = 0

    input_dict4 = {
        "input": input4,
        "scale": scale4,
        "zero_point": zero_point4,
        "quant_min": quant_min4,
        "quant_max": quant_max4,
        "ch_axis": ch_axis4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 4D input
    input5 = torch.randn(1, 3, 224, 224).numpy()
    scale5 = torch.abs(torch.randn(3)).numpy()
    zero_point5 = torch.randint(0, 256, (3,), dtype=torch.int32).numpy()
    quant_min5 = 0
    quant_max5 = 255
    ch_axis5 = 1

    input_dict5 = {
        "input": input5,
        "scale": scale5,
        "zero_point": zero_point5,
        "quant_min": quant_min5,
        "quant_max": quant_max5,
        "ch_axis": ch_axis5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.fake_quantize_per_channel_affine"] = fake_quantize_per_channel_affine_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.fake_quantize_per_channel_affine' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fake_quantize_per_channel_affine'.")

check_valid('torch.fake_quantize_per_channel_affine', generated_inputs['torch.fake_quantize_per_channel_affine'], lib="torch", suffix=0)
