
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def fake_quantize_per_channel_affine_inputs():
    list_of_inputs = []

    # Input 1: Basic float input with positive values
    input = torch.randn(1, 3, 4, 4).numpy()
    scale = torch.rand(3).numpy()
    zero_point = torch.randint(0, 256, (3,)).int().numpy()
    quant_min = 0
    quant_max = 255
    ch_axis = 1
    input_dict = {
        "input": input,
        "scale": scale,
        "zero_point": zero_point,
        "quant_min": quant_min,
        "quant_max": quant_max,
        "ch_axis": ch_axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float input with negative values
    input = torch.randn(1, 3, 4, 4).numpy()
    scale = torch.rand(3).numpy()
    zero_point = torch.randint(0, 256, (3,)).int().numpy()
    quant_min = 0
    quant_max = 255
    ch_axis = 1
    input_dict = {
        "input": input,
        "scale": scale,
        "zero_point": zero_point,
        "quant_min": quant_min,
        "quant_max": quant_max,
        "ch_axis": ch_axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Int input
    input = torch.randint(-100, 100, (1, 3, 4, 4)).numpy()
    scale = torch.rand(3).numpy()
    zero_point = torch.randint(0, 256, (3,)).int().numpy()
    quant_min = 0
    quant_max = 255
    ch_axis = 1
    input_dict = {
        "input": input,
        "scale": scale,
        "zero_point": zero_point,
        "quant_min": quant_min,
        "quant_max": quant_max,
        "ch_axis": ch_axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different channel axis
    input = torch.randn(3, 4, 4).numpy()
    scale = torch.rand(3).numpy()
    zero_point = torch.randint(0, 256, (3,)).int().numpy()
    quant_min = 0
    quant_max = 255
    ch_axis = 0
    input_dict = {
        "input": input,
        "scale": scale,
        "zero_point": zero_point,
        "quant_min": quant_min,
        "quant_max": quant_max,
        "ch_axis": ch_axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different quant min/max
    input = torch.randn(1, 3, 4, 4).numpy()
    scale = torch.rand(3).numpy()
    zero_point = torch.randint(0, 100, (3,)).int().numpy()
    quant_min = -128
    quant_max = 127
    ch_axis = 1
    input_dict = {
        "input": input,
        "scale": scale,
        "zero_point": zero_point,
        "quant_min": quant_min,
        "quant_max": quant_max,
        "ch_axis": ch_axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.fake_quantize_per_channel_affine"] = fake_quantize_per_channel_affine_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.fake_quantize_per_channel_affine' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fake_quantize_per_channel_affine'.")

check_valid('torch.fake_quantize_per_channel_affine', generated_inputs['torch.fake_quantize_per_channel_affine'], lib="torch")
