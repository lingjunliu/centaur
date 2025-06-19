
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def quantize_per_channel_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D tensor, axis=0
    input = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    scales = np.array([0.1, 0.2], dtype=np.float32)
    zero_points = np.array([10, 20], dtype=np.int64)
    axis = 0
    dtype = torch.quint8
    input_dict = {"input": input, "scales": scales, "zero_points": zero_points, "axis": axis, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor, axis=1
    input = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    scales = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    zero_points = np.array([10, 20, 30], dtype=np.int64)
    axis = 1
    dtype = torch.quint8
    input_dict = {"input": input, "scales": scales, "zero_points": zero_points, "axis": axis, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor, axis=0
    input = np.random.rand(2, 3, 4).astype(np.float32)
    scales = np.random.rand(2).astype(np.float32)
    zero_points = np.array([1, 2], dtype=np.int64)
    axis = 0
    dtype = torch.quint8
    input_dict = {"input": input, "scales": scales, "zero_points": zero_points, "axis": axis, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D tensor, axis=1
    input = np.random.rand(2, 3, 4).astype(np.float32)
    scales = np.random.rand(3).astype(np.float32)
    zero_points = np.array([1, 2, 3], dtype=np.int64)
    axis = 1
    dtype = torch.quint8
    input_dict = {"input": input, "scales": scales, "zero_points": zero_points, "axis": axis, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D tensor, axis=2
    input = np.random.rand(2, 3, 4).astype(np.float32)
    scales = np.random.rand(4).astype(np.float32)
    zero_points = np.array([1, 2, 3, 4], dtype=np.int64)
    axis = 2
    dtype = torch.quint8
    input_dict = {"input": input, "scales": scales, "zero_points": zero_points, "axis": axis, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.quantize_per_channel"] = quantize_per_channel_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.quantize_per_channel' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.quantize_per_channel'.")

check_valid('torch.quantize_per_channel', generated_inputs['torch.quantize_per_channel'], lib="torch", suffix=0)
