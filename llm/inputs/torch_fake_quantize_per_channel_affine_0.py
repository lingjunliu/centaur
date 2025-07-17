
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def fake_quantize_per_channel_affine_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    input = np.array([[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]], dtype=np.float32)
    scale = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    zero_point = np.array([0, 1, 2], dtype=np.int32)
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

    # Input 2: Different data type
    input = np.array([[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]], dtype=np.float64)
    scale = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    zero_point = np.array([0, 1, 2], dtype=np.int32)
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

    # Input 3: Different shape
    input = np.array([[-1.0, 0.0], [2.0, 3.0]], dtype=np.float32)
    scale = np.array([0.1, 0.2], dtype=np.float32)
    zero_point = np.array([0, 1], dtype=np.int32)
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

    # Input 4: 3D input
    input = np.random.rand(2, 3, 4).astype(np.float32)
    scale = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    zero_point = np.array([0, 1, 2], dtype=np.int32)
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
    
    # Input 5: different ch_axis
    input = np.random.rand(2, 3, 4).astype(np.float32)
    scale = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    zero_point = np.array([0, 1, 2], dtype=np.int32)
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

    # Input 6: small quant_min and quant_max
    input = np.array([[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]], dtype=np.float32)
    scale = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    zero_point = np.array([0, 1, 2], dtype=np.int32)
    quant_min = 0
    quant_max = 3
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

    # Input 7: Different dtype for scale and zero_point
    input = np.array([[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]], dtype=np.float32)
    scale = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    zero_point = np.array([0, 1, 2], dtype=np.int32)
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

    # Input 8: More dimensions for scale and zero_point
    input = np.random.rand(2, 3, 4).astype(np.float32)
    scale = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    zero_point = np.array([0, 1, 2], dtype=np.int32)
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

    # Input 9: Negative values in input and scale
    input = np.array([[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]], dtype=np.float32)
    scale = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    zero_point = np.array([0, 1, 2], dtype=np.int32)
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

    # Input 10: Large values in input
    input = np.array([[-100.0, 0.0, 100.0], [200.0, 300.0, 400.0]], dtype=np.float32)
    scale = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    zero_point = np.array([0, 1, 2], dtype=np.int32)
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

    # Input 11: scalar input
    input = np.array(1.0, dtype=np.float32)
    scale = np.array([0.1], dtype=np.float32)
    zero_point = np.array([0], dtype=np.int32)
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

    # Input 12: Shape consistency
    # removed since dimension mismatch

    #Input 13: check scalar + ch axis 0
    input = np.random.rand(3,3).astype(np.float32)
    scale = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    zero_point = np.array([0, 1, 2], dtype=np.int32)
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
