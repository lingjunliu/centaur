
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def quantize_per_tensor_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    scale = 0.5
    zero_point = 10
    dtype = torch.int8  # Changed back to int8 and using qint8/quint8 appropriately
    input_dict = {"input": input_tensor, "scale": scale, "zero_point": zero_point, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    scale = 0.25
    zero_point = -5
    dtype = torch.int8
    input_dict = {"input": input_tensor, "scale": scale, "zero_point": zero_point, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    scale = 1.0
    zero_point = 0
    dtype = torch.int8
    input_dict = {"input": input_tensor, "scale": scale, "zero_point": zero_point, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    scale = 0.1
    zero_point = 0
    dtype = torch.int8
    input_dict = {"input": input_tensor, "scale": scale, "zero_point": zero_point, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float64)
    scale = 0.05
    zero_point = 0
    dtype = torch.int8
    input_dict = {"input": input_tensor, "scale": scale, "zero_point": zero_point, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float16)
    scale = 0.75
    zero_point = 0
    dtype = torch.int8
    input_dict = {"input": input_tensor, "scale": scale, "zero_point": zero_point, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([-1.0, -2.0, -3.0, -4.0, -5.0], dtype=np.float16)
    scale = 0.3
    zero_point = 0
    dtype = torch.int8
    input_dict = {"input": input_tensor, "scale": scale, "zero_point": zero_point, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    scale = 0.2
    zero_point = 0
    dtype = torch.int8
    input_dict = {"input": input_tensor, "scale": scale, "zero_point": zero_point, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    scale = 0.1
    zero_point = 0
    dtype = torch.int8
    input_dict = {"input": input_tensor, "scale": scale, "zero_point": zero_point, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([[-0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    scale = 0.6
    zero_point = 0
    dtype = torch.int8
    input_dict = {"input": input_tensor, "scale": scale, "zero_point": zero_point, "dtype": dtype}
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
