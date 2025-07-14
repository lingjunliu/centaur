
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def quantize_per_tensor_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    scale = 0.5
    zero_point = 10
    dtype = torch.int8
    input_dict = {"input": input_tensor.numpy(), "scale": scale, "zero_point": zero_point, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.float32)
    scale = 0.25
    zero_point = -5
    dtype = torch.int8
    input_dict = {"input": input_tensor.numpy(), "scale": scale, "zero_point": zero_point, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    scale = 1.0
    zero_point = 0
    dtype = torch.uint8
    input_dict = {"input": input_tensor.numpy(), "scale": scale, "zero_point": zero_point, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float64)
    scale = 0.01
    zero_point = 128
    dtype = torch.int16
    input_dict = {"input": input_tensor.numpy(), "scale": scale, "zero_point": zero_point, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = torch.tensor([-0.1, -0.2, -0.3, -0.4, -0.5], dtype=torch.float64)
    scale = 0.005
    zero_point = -128
    dtype = torch.int16
    input_dict = {"input": input_tensor.numpy(), "scale": scale, "zero_point": zero_point, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    scale = 0.75
    zero_point = 100
    dtype = torch.int32
    input_dict = {"input": input_tensor.numpy(), "scale": scale, "zero_point": zero_point, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.float32)
    scale = 0.125
    zero_point = -100
    dtype = torch.int32
    input_dict = {"input": input_tensor.numpy(), "scale": scale, "zero_point": zero_point, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(2, 3).to(torch.float32)
    scale = 0.3
    zero_point = 50
    dtype = torch.uint8
    input_dict = {"input": input_tensor.numpy(), "scale": scale, "zero_point": zero_point, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]).reshape(2, 3).to(torch.float32)
    scale = 0.07
    zero_point = -50
    dtype = torch.int8
    input_dict = {"input": input_tensor.numpy(), "scale": scale, "zero_point": zero_point, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(1, 2, 3).to(torch.float32)
    scale = 0.8
    zero_point = 20
    dtype = torch.int8
    input_dict = {"input": input_tensor.numpy(), "scale": scale, "zero_point": zero_point, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    input_tensor = torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]).reshape(1, 2, 3).to(torch.float32)
    scale = 0.03
    zero_point = -15
    dtype = torch.int16
    input_dict = {"input": input_tensor.numpy(), "scale": scale, "zero_point": zero_point, "dtype": dtype}
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
