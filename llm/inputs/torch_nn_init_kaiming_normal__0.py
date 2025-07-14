
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def kaiming_normal_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D tensor, fan_in mode, ReLU nonlinearity
    tensor = np.zeros((2, 3), dtype=np.float32)
    a = 0.0
    mode = "fan_in"
    nonlinearity = "relu"

    input_dict = {
        "tensor": tensor,
        "a": a,
        "mode": mode,
        "nonlinearity": nonlinearity
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor, fan_out mode, LeakyReLU nonlinearity
    tensor = np.zeros((2, 3), dtype=np.float64)
    a = 0.1
    mode = "fan_out"
    nonlinearity = "leaky_relu"

    input_dict = {
        "tensor": tensor,
        "a": a,
        "mode": mode,
        "nonlinearity": nonlinearity
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor, fan_in mode, linear nonlinearity
    tensor = np.zeros((2, 3, 4), dtype=np.float32)
    a = 0.0
    mode = "fan_in"
    nonlinearity = "linear"

    input_dict = {
        "tensor": tensor,
        "a": a,
        "mode": mode,
        "nonlinearity": nonlinearity
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D tensor, fan_out mode, sigmoid nonlinearity
    tensor = np.zeros((2, 3, 4, 5), dtype=np.float64)
    a = 0.0
    mode = "fan_out"
    nonlinearity = "sigmoid"

    input_dict = {
        "tensor": tensor,
        "a": a,
        "mode": mode,
        "nonlinearity": nonlinearity
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 2D tensor, fan_in mode, tanh nonlinearity, a = 1.0
    tensor = np.zeros((3, 5), dtype=np.float32)
    a = 1.0
    mode = "fan_in"
    nonlinearity = "tanh"

    input_dict = {
        "tensor": tensor,
        "a": a,
        "mode": mode,
        "nonlinearity": nonlinearity
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D tensor, fan_out mode, ReLU nonlinearity
    tensor = np.zeros((5, 5), dtype=np.float64)
    a = 0.0
    mode = "fan_out"
    nonlinearity = "relu"

    input_dict = {
        "tensor": tensor,
        "a": a,
        "mode": mode,
        "nonlinearity": nonlinearity
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D tensor, fan_in mode, leaky_relu nonlinearity with small negative slope
    tensor = np.zeros((3, 5, 2), dtype=np.float32)
    a = 0.01
    mode = "fan_in"
    nonlinearity = "leaky_relu"

    input_dict = {
        "tensor": tensor,
        "a": a,
        "mode": mode,
        "nonlinearity": nonlinearity
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D tensor, fan_out mode, linear nonlinearity, different a
    tensor = np.zeros((1, 2, 3, 4), dtype=np.float64)
    a = 0.5
    mode = "fan_out"
    nonlinearity = "linear"

    input_dict = {
        "tensor": tensor,
        "a": a,
        "mode": mode,
        "nonlinearity": nonlinearity
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D tensor, fan_in mode, sigmoid nonlinearity, a = 0.0
    tensor = np.zeros((8, 2), dtype=np.float32)
    a = 0.0
    mode = "fan_in"
    nonlinearity = "sigmoid"

    input_dict = {
        "tensor": tensor,
        "a": a,
        "mode": mode,
        "nonlinearity": nonlinearity
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 2D tensor, fan_out mode, tanh nonlinearity, a = 2.0
    tensor = np.zeros((4, 7), dtype=np.float64)
    a = 2.0
    mode = "fan_out"
    nonlinearity = "tanh"

    input_dict = {
        "tensor": tensor,
        "a": a,
        "mode": mode,
        "nonlinearity": nonlinearity
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: 5D tensor, fan_in mode, ReLU nonlinearity
    tensor = np.zeros((2, 3, 4, 5, 6), dtype=np.float32)
    a = 0.0
    mode = "fan_in"
    nonlinearity = "relu"

    input_dict = {
        "tensor": tensor,
        "a": a,
        "mode": mode,
        "nonlinearity": nonlinearity
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: 3D tensor, fan_out mode, leaky_relu nonlinearity with small negative slope, different a
    tensor = np.zeros((4, 6, 3), dtype=np.float64)
    a = 0.2
    mode = "fan_out"
    nonlinearity = "leaky_relu"

    input_dict = {
        "tensor": tensor,
        "a": a,
        "mode": mode,
        "nonlinearity": nonlinearity
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.init.kaiming_normal_"] = kaiming_normal_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.init.kaiming_normal_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.init.kaiming_normal_'.")

check_valid('torch.nn.init.kaiming_normal_', generated_inputs['torch.nn.init.kaiming_normal_'], lib="torch", suffix=0)
