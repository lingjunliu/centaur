
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def fake_quantize_per_channel_affine_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]], dtype=np.float32)
    scale_tensor = np.array([0.1, 0.2], dtype=np.float32)
    zero_point_tensor = np.array([0, 1], dtype=np.int32)
    quant_min_val = 0
    quant_max_val = 255
    ch_axis_val = 0
    
    input_dict = {
        "input": input_tensor,
        "scale": scale_tensor,
        "zero_point": zero_point_tensor,
        "quant_min": quant_min_val,
        "quant_max": quant_max_val,
        "ch_axis": ch_axis_val
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[-1.0, 0.0], [2.0, 3.0]], dtype=np.float32)
    scale_tensor = np.array([0.1, 0.2], dtype=np.float32)
    zero_point_tensor = np.array([0, 1], dtype=np.int32)
    quant_min_val = -128
    quant_max_val = 127
    ch_axis_val = 1
    
    input_dict = {
        "input": input_tensor,
        "scale": scale_tensor,
        "zero_point": zero_point_tensor,
        "quant_min": quant_min_val,
        "quant_max": quant_max_val,
        "ch_axis": ch_axis_val
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.randn(2, 3, 4).astype(np.float32)
    scale_tensor = np.abs(np.random.randn(3).astype(np.float32))
    zero_point_tensor = np.random.randint(0, 255, size=(3,), dtype=np.int32)
    quant_min_val = 0
    quant_max_val = 255
    ch_axis_val = 1
    
    input_dict = {
        "input": input_tensor,
        "scale": scale_tensor,
        "zero_point": zero_point_tensor,
        "quant_min": quant_min_val,
        "quant_max": quant_max_val,
        "ch_axis": ch_axis_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.random.randn(1, 5, 1).astype(np.float32)
    scale_tensor = np.abs(np.random.randn(5).astype(np.float32))
    zero_point_tensor = np.random.randint(-128, 127, size=(5,), dtype=np.int32)
    quant_min_val = -128
    quant_max_val = 127
    ch_axis_val = 1
    
    input_dict = {
        "input": input_tensor,
        "scale": scale_tensor,
        "zero_point": zero_point_tensor,
        "quant_min": quant_min_val,
        "quant_max": quant_max_val,
        "ch_axis": ch_axis_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    scale_tensor = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    zero_point_tensor = np.array([0, 1, 2], dtype=np.int32)
    quant_min_val = 0
    quant_max_val = 255
    ch_axis_val = 0
    
    input_dict = {
        "input": input_tensor,
        "scale": scale_tensor,
        "zero_point": zero_point_tensor,
        "quant_min": quant_min_val,
        "quant_max": quant_max_val,
        "ch_axis": ch_axis_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.random.randn(3, 4, 5).astype(np.float32)
    scale_tensor = np.abs(np.random.randn(4).astype(np.float32))
    zero_point_tensor = np.random.randint(0, 255, size=(4,), dtype=np.int32)
    quant_min_val = 0
    quant_max_val = 255
    ch_axis_val = 1
    
    input_dict = {
        "input": input_tensor,
        "scale": scale_tensor,
        "zero_point": zero_point_tensor,
        "quant_min": quant_min_val,
        "quant_max": quant_max_val,
        "ch_axis": ch_axis_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.random.randn(4, 2).astype(np.float32)
    scale_tensor = np.abs(np.random.randn(2).astype(np.float32))
    zero_point_tensor = np.random.randint(-128, 127, size=(2,), dtype=np.int32)
    quant_min_val = -128
    quant_max_val = 127
    ch_axis_val = 0
    
    input_dict = {
        "input": input_tensor,
        "scale": scale_tensor,
        "zero_point": zero_point_tensor,
        "quant_min": quant_min_val,
        "quant_max": quant_max_val,
        "ch_axis": ch_axis_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input_tensor = np.array([1.0, 2.0, 3.0, 4.0]).reshape(1,4).astype(np.float32)
    scale_tensor = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    zero_point_tensor = np.array([0, 1, 2, 3], dtype=np.int32)
    quant_min_val = 0
    quant_max_val = 255
    ch_axis_val = 1
    
    input_dict = {
        "input": input_tensor,
        "scale": scale_tensor,
        "zero_point": zero_point_tensor,
        "quant_min": quant_min_val,
        "quant_max": quant_max_val,
        "ch_axis": ch_axis_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    

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
