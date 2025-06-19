
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def quantize_per_tensor_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D tensor, positive values
    input_tensor = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    scale = np.float32(0.5)
    zero_point = np.int32(0)
    dtype = torch.qint8

    input_dict = {
        "input": input_tensor,
        "scale": scale,
        "zero_point": zero_point,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor, mixed positive and negative values
    input_tensor = np.array([[-1.0, 0.0], [2.0, -3.0]], dtype=np.float32)
    scale = np.float32(0.25)
    zero_point = np.int32(10)
    dtype = torch.qint8

    input_dict = {
        "input": input_tensor,
        "scale": scale,
        "zero_point": zero_point,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor, smaller scale
    input_tensor = np.random.rand(2, 2, 2).astype(np.float32)
    scale = np.float32(0.01)
    zero_point = np.int32(128)
    dtype = torch.quint8

    input_dict = {
        "input": input_tensor,
        "scale": scale,
        "zero_point": zero_point,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D tensor, larger zero_point
    input_tensor = np.array([-5.0, -2.0, 0.0, 3.0], dtype=np.float32)
    scale = np.float32(0.1)
    zero_point = np.int32(-10)
    dtype = torch.quint8

    input_dict = {
        "input": input_tensor,
        "scale": scale,
        "zero_point": zero_point,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.quantize_per_tensor"] = quantize_per_tensor_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.quantize_per_tensor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.quantize_per_tensor'.")

check_valid('torch.quantize_per_tensor', generated_inputs['torch.quantize_per_tensor'], lib="torch", suffix=0)
