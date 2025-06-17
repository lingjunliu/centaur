
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def quantize_per_channel_inputs():
    list_of_inputs = []

    # Input 1: Basic float input
    input = np.array([[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]], dtype=np.float32)
    scales = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    zero_points = np.array([0, 1, 2], dtype=np.int32)
    axis = 1
    dtype = torch.float32

    input_dict = {
        "input": input,
        "scales": scales,
        "zero_points": zero_points,
        "axis": axis,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Int input
    input = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    scales = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    zero_points = np.array([0, 1, 2], dtype=np.int32)
    axis = 1
    dtype = torch.float32

    input_dict = {
        "input": input,
        "scales": scales,
        "zero_points": zero_points,
        "axis": axis,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D input
    input = np.random.rand(2, 3, 4).astype(np.float32)
    scales = np.random.rand(3).astype(np.float32)
    zero_points = np.array([0, 1, 2], dtype=np.int32)
    axis = 1
    dtype = torch.float32

    input_dict = {
        "input": input,
        "scales": scales,
        "zero_points": zero_points,
        "axis": axis,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.quantize_per_channel"] = quantize_per_channel_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.quantize_per_channel' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.quantize_per_channel'.")

check_valid('torch.quantize_per_channel', generated_inputs['torch.quantize_per_channel'], lib="torch")
